"""task.md 的解析与改写。

一个文件容纳所有 tmux 窗口的队列，格式如下::

    # tmux: 0                  <- 窗口（tmux 会话名，或 "会话:窗口" 形式的 target）
    workdir: /path/to/project  <- 命令在此目录执行（默认当前目录）

    ## task-list              <- 每行一条待执行命令；以 # 开头的行是注释
    bash train_a.sh --batch_size 8   <- CUDA OOM 重试时，命令里的 --batch_size 会被逐次减半

    ## worked list            <- 已启动的命令由程序自动移到这里
"""
import fcntl
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

WINDOW_RE = re.compile(r"^#\s*tmux:\s*(.+?)\s*$")
TASK_LIST_RE = re.compile(r"^##\s*task-list\s*$", re.I)
WORKED_RE = re.compile(r"^##\s*worked list\s*$", re.I)
KEY_RE = re.compile(r"^(workdir)\s*:\s*(.+?)\s*$")


@dataclass
class Window:
    name: str
    workdir: str
    tasks: List[str]  # task-list 中尚未执行的原始行（已去掉空行和注释）


def _is_task(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _sections(lines: List[str]) -> Dict[str, Dict[str, object]]:
    """返回 {窗口名: {"start": 标题行号, "end": 下一个窗口标题行号, "task": (起,止), "worked": (起,止)|None}}。

    task/worked 的 (起,止) 为正文行区间（左闭右开），不含 ## 标题行。
    """
    heads = [(i, m.group(1)) for i, l in enumerate(lines) if (m := WINDOW_RE.match(l))]
    out: Dict[str, Dict[str, object]] = {}
    for n, (start, name) in enumerate(heads):
        if name in out:
            raise ValueError(f"窗口 {name!r} 在 task 文件中重复出现（第 {start + 1} 行）")
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        subs = [(i, "task") if TASK_LIST_RE.match(lines[i]) else (i, "worked")
                for i in range(start + 1, end)
                if TASK_LIST_RE.match(lines[i]) or WORKED_RE.match(lines[i])]
        sec: Dict[str, object] = {"start": start, "end": end, "task": None, "worked": None}
        for k, (i, kind) in enumerate(subs):
            stop = subs[k + 1][0] if k + 1 < len(subs) else end
            sec[kind] = (i + 1, stop)
        out[name] = sec
    return out


def load_windows(path: Path) -> Dict[str, Window]:
    lines = Path(path).read_text().splitlines()
    windows: Dict[str, Window] = {}
    for name, sec in _sections(lines).items():
        first_sub = min([r[0] - 1 for r in (sec["task"], sec["worked"]) if r] or [sec["end"]])
        conf: Dict[str, str] = {}
        for l in lines[sec["start"] + 1:first_sub]:
            if (m := KEY_RE.match(l.strip())):
                conf[m.group(1)] = m.group(2)
        a, b = sec["task"] or (0, 0)
        windows[name] = Window(
            name=name,
            workdir=conf.get("workdir", str(Path.cwd())),
            tasks=[l.strip() for l in lines[a:b] if _is_task(l)],
        )
    return windows


@contextmanager
def _locked(path: Path):
    # task.md 可能被多个守护进程（不同窗口）同时改写，用独立锁文件串行化。
    with open(str(path) + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _write(path: Path, lines: List[str]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)  # 原子替换，避免用户/其他进程读到写了一半的文件


def move_first_task_to_worked(path: Path, name: str, expected: str) -> bool:
    """把窗口队列的第一条命令移到 worked list；若第一条已不是 expected 则不改动并返回 False。"""
    path = Path(path)
    with _locked(path):
        lines = path.read_text().splitlines()
        sec = _sections(lines).get(name)
        if not sec or not sec["task"]:
            return False
        a, b = sec["task"]
        idx = next((i for i in range(a, b) if _is_task(lines[i])), None)
        if idx is None or lines[idx].strip() != expected:
            return False
        moved = lines.pop(idx).strip()
        # 删除一行后，后面的区间整体前移 1
        sec = _sections(lines)[name]
        if sec["worked"]:
            # 追加在 worked 区间最后一个非空行之后
            wa, wb = sec["worked"]
            at = wb
            while at > wa and not lines[at - 1].strip():
                at -= 1
            lines.insert(at, moved)
        else:
            at = sec["end"]
            while at > sec["start"] + 1 and not lines[at - 1].strip():
                at -= 1
            lines[at:at] = ["", "## worked list", moved]
        _write(path, lines)
        return True
