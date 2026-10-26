"""对 tmux 命令行的薄封装。"""
import subprocess
import time
from typing import List, Optional


class Tmux:
    def __init__(self, socket: Optional[str] = None):
        # socket 即 `tmux -L <name>`，用于测试时隔离出独立的 tmux server
        self.base = ["tmux"] + (["-L", socket] if socket else [])

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(self.base + list(args), capture_output=True, text=True)

    @staticmethod
    def target(name: str) -> str:
        # "会话:窗口" 原样使用；纯会话名加 "=" 前缀做精确匹配，避免 "0" 被当成窗口序号、"sv" 被当成前缀。
        return name if ":" in name else f"={name}:"

    def pane_pid(self, name: str) -> Optional[int]:
        """窗口当前 pane 的 shell 进程号；窗口/会话不存在时返回 None。"""
        r = self._run("display-message", "-p", "-t", self.target(name), "#{pane_pid}")
        out = r.stdout.strip()
        return int(out) if r.returncode == 0 and out.isdigit() else None

    @staticmethod
    def busy(pane_pid: int) -> bool:
        """pane 的 shell 下是否有子进程，即前台是否有命令在跑。"""
        return subprocess.run(["pgrep", "-P", str(pane_pid)], capture_output=True).returncode == 0

    def wait_idle(self, pane_pid: int, timeout: float) -> bool:
        end = time.time() + timeout
        while self.busy(pane_pid):
            if time.time() >= end:
                return False
            time.sleep(0.2)
        return True

    def send(self, name: str, text: str) -> None:
        t = self.target(name)
        self._run("send-keys", "-t", t, "-l", "--", text)
        self._run("send-keys", "-t", t, "C-m")

    def sessions(self) -> List[str]:
        r = self._run("list-sessions", "-F", "#{session_name}")
        return r.stdout.split() if r.returncode == 0 else []
