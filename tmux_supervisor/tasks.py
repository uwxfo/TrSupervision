"""task.md 的解析与改写。

一个文件容纳所有 tmux 窗口的队列，格式如下::

    # tmux: 0                  <- 窗口（tmux 会话名，或 "会话:窗口" 形式的 target）
    workdir: /path/to/project  <- 命令在此目录执行（默认当前目录）

    ## task-list              <- 每行一条待执行命令；以 # 开头的行是注释
    bash train_a.sh --batch_size 8   <- CUDA OOM 重试时，命令里的 --batch_size 会被逐次减半
    [workdir=sub/dir] bash b.sh      <- 只对这一条改变工作目录（相对路径相对窗口的 workdir）

    ## worked list           <- 成功结束的命令，由程序从 task-list 移入，行尾带 `# [done log=... output_dir=...]`
    ## failed list           <- 重试用完 / shell 语法错误的命令，行尾带 `# [failed 原因]`，等待人工处理

运行中的命令一直留在 task-list 里，成功后才移走。
"""
import fcntl
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WINDOW_RE = re.compile(r"^#\s*tmux:\s*(.+?)\s*$")
SUB_RES = {
    "task": re.compile(r"^##\s*task-list\s*$", re.I),
    "worked": re.compile(r"^##\s*worked list\s*$", re.I),
    "failed": re.compile(r"^##\s*failed list\s*$", re.I),
}
TITLES = {"task": "task-list", "worked": "worked list", "failed": "failed list"}
KEY_RE = re.compile(r"^(workdir)\s*:\s*(.+?)\s*$")
TASK_WORKDIR_RE = re.compile(r"^\[workdir=([^\]]+)\]\s*(.+)$")
# 只剥掉本程序写入的固定格式标记（failed / done），不碰用户自己的 `# 注释`（命令里的 # 可能在引号内）
FAILED_MARK_RE = re.compile(r"\s+# \[(?:failed|done)[^\]]*\]\s*$")


@dataclass
class Window:
    name: str
    workdir: str
    tasks: List[str]   # task-list 中尚未成功的原始行（已去掉空行和注释）
    worked: List[str]
    failed: List[str]


def _is_task(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _sections(lines: List[str]) -> Dict[str, Dict[str, object]]:
    """返回 {窗口名: {"start": 标题行号, "end": 下一个窗口标题行号, "task"/"worked"/"failed": (起,止)|None}}。

    (起,止) 为正文行区间（左闭右开），不含 ## 标题行。
    """
    heads = [(i, m.group(1)) for i, l in enumerate(lines) if (m := WINDOW_RE.match(l))]
    out: Dict[str, Dict[str, object]] = {}
    for n, (start, name) in enumerate(heads):
        if name in out:
            raise ValueError(f"window {name!r} appears more than once in the task file (line {start + 1})")
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        subs = [(i, kind) for i in range(start + 1, end) for kind, rx in SUB_RES.items() if rx.match(lines[i])]
        sec: Dict[str, object] = {"start": start, "end": end, "task": None, "worked": None, "failed": None}
        for k, (i, kind) in enumerate(subs):
            stop = subs[k + 1][0] if k + 1 < len(subs) else end
            sec[kind] = (i + 1, stop)
        out[name] = sec
    return out


def load_windows(path: Path) -> Dict[str, Window]:
    lines = Path(path).read_text().splitlines()
    windows: Dict[str, Window] = {}
    for name, sec in _sections(lines).items():
        ranges = {k: sec[k] or (0, 0) for k in ("task", "worked", "failed")}
        first_sub = min([sec[k][0] - 1 for k in ("task", "worked", "failed") if sec[k]] or [sec["end"]])
        conf: Dict[str, str] = {}
        for l in lines[sec["start"] + 1:first_sub]:
            if (m := KEY_RE.match(l.strip())):
                conf[m.group(1)] = m.group(2)
        body = {k: [l.strip() for l in lines[a:b] if _is_task(l)] for k, (a, b) in ranges.items()}
        windows[name] = Window(name=name, workdir=conf.get("workdir", str(Path.cwd())),
                               tasks=body["task"], worked=body["worked"], failed=body["failed"])
    return windows


def parse_task(line: str, window_workdir: str) -> Tuple[str, str]:
    """把 task-list 的一行拆成 (命令, 工作目录)。"""
    line = FAILED_MARK_RE.sub("", line.strip())
    m = TASK_WORKDIR_RE.match(line)
    if not m:
        return line, window_workdir
    return m.group(2).strip(), str(Path(window_workdir) / m.group(1).strip())  # 绝对路径时 Path 忽略前缀


@contextmanager
def _locked(path: Path):
    # task.md 可能被多个监控进程（不同窗口）同时改写，用独立锁文件串行化。
    with open(str(path) + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _write(path: Path, lines: List[str]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)  # 原子替换，避免用户/其他进程读到写了一半的文件


def move_task(path: Path, name: str, raw_line: str, target: str, note: Optional[str] = None) -> bool:
    """把窗口 task-list 里内容等于 raw_line 的第一行移到 target（"worked" 或 "failed"）列表末尾。

    按内容匹配而不是按行号：运行期间你可能在前面插入了新行。failed 总是在行尾附 `# [failed note]`，worked 在给了 note 时附 `# [done note]`。
    找不到该行（已被你删除/改动）时不改动文件并返回 False。
    """
    path = Path(path)
    with _locked(path):
        lines = path.read_text().splitlines()
        sec = _sections(lines).get(name)
        if not sec or not sec["task"]:
            return False
        a, b = sec["task"]
        idx = next((i for i in range(a, b) if _is_task(lines[i]) and lines[i].strip() == raw_line), None)
        if idx is None:
            return False
        moved = lines.pop(idx).strip()
        if target == "failed" or note:  # 在行尾附上程序写的标记：failed 原因 / done 的日志与输出位置
            mark = {"worked": "done", "failed": "failed"}[target]
            moved = FAILED_MARK_RE.sub("", moved) + f"  # [{mark} {(note or '').replace(']', ')')}]"
        sec = _sections(lines)[name]  # 删除一行后，后面的区间整体前移了 1，重新计算
        if sec[target]:
            wa, wb = sec[target]
            at = wb
            while at > wa and not lines[at - 1].strip():  # 追加在该区间最后一个非空行之后
                at -= 1
            lines.insert(at, moved)
        else:
            at = sec["end"]
            while at > sec["start"] + 1 and not lines[at - 1].strip():
                at -= 1
            lines[at:at] = ["", f"## {TITLES[target]}", moved]
        _write(path, lines)
        return True
