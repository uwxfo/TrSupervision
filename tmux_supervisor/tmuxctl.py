"""对 tmux 命令行的薄封装。"""
import re
import subprocess
import time
from typing import List, Optional

# 这些进程只是把终端转接到别处的 shell（计算节点 / 远程主机），本机看不到里面的子进程
WRAPPERS = {"srun", "salloc", "ssh"}
# 光标所在行以 shell 提示符结尾（$ # % >）才算空闲；训练日志一般不会以这些符号结尾
PROMPT_RE = re.compile(r"[$#%>]\s*$")


class TmuxError(Exception):
    """tmux 命令超时没有响应。"""


class Tmux:
    def __init__(self, socket: Optional[str] = None, timeout: float = 15, binary: str = "tmux"):
        # socket 即 `tmux -L <name>`，用于测试时隔离出独立的 tmux server
        self.base = [binary] + (["-L", socket] if socket else [])
        self.timeout = timeout  # 每次调用 tmux / pgrep 的超时秒数；曾出现 send-keys 卡住 17 小时，不设超时监控会无声冻结

    def _call(self, cmd: List[str]) -> subprocess.CompletedProcess:
        try:
            # stdin 接 /dev/null：避免子进程继承监控进程所在的终端；超时后 run 会杀掉子进程
            return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise TmuxError(f"`{' '.join(cmd[:4])}` did not respond within {self.timeout:g}s")

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return self._call(self.base + list(args))

    @staticmethod
    def target(name: str) -> str:
        # "会话:窗口" 原样使用；纯会话名加 "=" 前缀做精确匹配，避免 "0" 被当成窗口序号、"sv" 被当成前缀。
        return name if ":" in name else f"={name}:"

    def pane_pid(self, name: str) -> Optional[int]:
        """窗口当前 pane 的 shell 进程号；窗口/会话不存在时返回 None。"""
        r = self._run("display-message", "-p", "-t", self.target(name), "#{pane_pid}")
        out = r.stdout.strip()
        return int(out) if r.returncode == 0 and out.isdigit() else None

    def children(self, pane_pid: int) -> List[str]:
        """pane 的 shell 下各子进程的进程名（comm）。"""
        return self._call(["ps", "-o", "comm=", "--ppid", str(pane_pid)]).stdout.split()

    def busy(self, pane_pid: int, name: Optional[str] = None) -> bool:
        """前台是否有命令在跑。

        pane 的 shell 下有普通子进程即忙。若子进程全是 srun/salloc/ssh 这类转接进程（嵌套 shell，
        例如 `srun --pty bash` 进计算节点），本机看不到里面是否在跑命令，改看光标所在行是不是 shell 提示符
        （需要传窗口名 name；不传则保守地认为忙）。
        """
        kids = self.children(pane_pid)
        if not kids:
            return False
        if name is None or any(k not in WRAPPERS for k in kids):
            return True
        return not self.at_prompt(name)

    def at_prompt(self, name: str) -> bool:
        t = self.target(name)
        y = self._run("display-message", "-p", "-t", t, "#{cursor_y}").stdout.strip()
        if not y.isdigit():
            return False
        line = self._run("capture-pane", "-p", "-t", t, "-S", y, "-E", y).stdout
        return bool(PROMPT_RE.search(line.rstrip("\n")))

    def wait_idle(self, pane_pid: int, timeout: float, name: Optional[str] = None) -> bool:
        end = time.time() + timeout
        while self.busy(pane_pid, name):
            if time.time() >= end:
                return False
            time.sleep(0.2)
        return True

    def _leave_copy_mode(self, t: str) -> None:
        # 窗口处于 copy-mode（滚动查看历史）时按键会被模式吞掉，命令发不进 shell，先退出该模式
        if self._run("display-message", "-p", "-t", t, "#{pane_in_mode}").stdout.strip() == "1":
            self._run("send-keys", "-t", t, "-X", "cancel")

    def clear_input(self, name: str, pane_pid: int) -> bool:
        """发命令前清掉 shell 输入行上的残留（没发出的半行、续行提示符 `>`、Ctrl-R 搜索等）：在空闲 shell 里发 C-c 即可全部取消。

        C-c 会打断前台程序，所以只在刚确认窗口空闲时才发；有程序在跑时什么也不发并返回 False。
        嵌套 shell（srun/salloc/ssh）里的"空闲"只是看提示符的推断，程序输出恰好以 $#%> 结尾会误判，
        所以那里不发 C-c 而发 C-a C-k（行首 + 删到行尾）：只清残留的半行，误判时也只是几个控制字符，不会杀掉程序。
        """
        if self.busy(pane_pid, name):
            return False
        t = self.target(name)
        self._leave_copy_mode(t)
        if self.nested(pane_pid):  # 判为空闲却还有子进程：只能是转接进程
            self.clear_line(name)
        else:
            self._run("send-keys", "-t", t, "C-c")
        return True

    def nested(self, pane_pid: int) -> bool:
        """窗口里只有 srun/salloc/ssh 这类转接进程，即身处嵌套 shell。"""
        kids = self.children(pane_pid)
        return bool(kids) and all(k in WRAPPERS for k in kids)

    def clear_line(self, name: str) -> None:
        """C-a C-k：行首 + 删到行尾，清掉没发出的半行。是普通控制字符、不是信号，发给正在跑的程序也不会杀掉它。"""
        t = self.target(name)
        self._leave_copy_mode(t)
        self._run("send-keys", "-t", t, "C-a", "C-k")

    def send(self, name: str, text: str) -> None:
        t = self.target(name)
        self._leave_copy_mode(t)
        self._run("send-keys", "-t", t, "-l", "--", text)
        self._run("send-keys", "-t", t, "C-m")

    def sessions(self) -> List[str]:
        r = self._run("list-sessions", "-F", "#{session_name}")
        return r.stdout.split() if r.returncode == 0 else []
