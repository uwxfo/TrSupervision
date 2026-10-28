"""对 tmux 命令行的薄封装。"""
import subprocess
import time
from typing import List, Optional


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

    def busy(self, pane_pid: int) -> bool:
        """pane 的 shell 下是否有子进程，即前台是否有命令在跑。"""
        return self._call(["pgrep", "-P", str(pane_pid)]).returncode == 0

    def wait_idle(self, pane_pid: int, timeout: float) -> bool:
        end = time.time() + timeout
        while self.busy(pane_pid):
            if time.time() >= end:
                return False
            time.sleep(0.2)
        return True

    def clear_input(self, name: str, pane_pid: int) -> bool:
        """发命令前清掉 shell 输入行上的残留（没发出的半行、续行提示符 `>`、Ctrl-R 搜索等）：在空闲 shell 里发 C-c 即可全部取消。

        C-c 会打断前台程序，所以只在刚确认窗口空闲时才发；有程序在跑时什么也不发并返回 False。
        """
        if self.busy(pane_pid):
            return False
        self._run("send-keys", "-t", self.target(name), "C-c")
        return True

    def send(self, name: str, text: str) -> None:
        t = self.target(name)
        self._run("send-keys", "-t", t, "-l", "--", text)
        self._run("send-keys", "-t", t, "C-m")

    def sessions(self) -> List[str]:
        r = self._run("list-sessions", "-F", "#{session_name}")
        return r.stdout.split() if r.returncode == 0 else []
