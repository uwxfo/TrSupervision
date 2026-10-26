"""核心监督逻辑：对每个 tmux 窗口执行 检查 -> (重试 | 暂停 | 启动队列下一条)。

任务通过 send-keys 发到窗口的交互 shell 里执行，包装命令在结束时把退出码写入
`<state_dir>/<窗口>.<run_id>.exit`，监督器据此判断成功/失败，不依赖解析屏幕输出。
"""
import fcntl
import json
import re
import shlex
import subprocess
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .tasks import Window, load_windows, move_first_task_to_worked
from .tmuxctl import Tmux

# 只匹配 CUDA 显存不足；"illegal memory access" 之类不是 OOM，减小 batch 没用。
OOM_RE = re.compile(r"CUDA out of memory|CUDA error:.*out of memory|CUBLAS_STATUS_ALLOC_FAILED"
                    r"|CUDNN_STATUS_ALLOC_FAILED", re.I)
# 命令行里的 batch_size 参数（支持 `--batch_size 4` 与 `--batch_size=4`），OOM 重试时改写它
BATCH_ARG_RE = re.compile(r"(--batch_size[ =])(\d+)")
# 训练程序启动时会 print(args)，其中的 output_dir 即本次运行实际写 checkpoint 的目录（相对任务的 workdir）
OUTPUT_DIR_RE = re.compile(r"output_dir='([^']+)'")
RESUME_ARG_RE = re.compile(r"--resume(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|\S+)")
INTERRUPT_RCS = (130, 143)  # Ctrl-C / SIGTERM：视为人工中断，暂停而不重试
START_GRACE = 15  # 发出命令后，等多久才允许判定"窗口空闲却没有退出记录"（shell 启动子进程需要时间）
LOG_TAIL_BYTES = 200_000  # 判断错误类型时只读日志末尾；找 output_dir 时读日志开头同样大小


def safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def find_checkpoint(out_dir: Path) -> Optional[Path]:
    """返回目录里最新的完好 checkpoint：先 checkpoint.pth（每个 epoch 覆盖写），再按编号从大到小的 checkpointNNNN.pth。

    torch.save 写的是 zip，被中途打断（如磁盘写满）的文件缺少文件尾的目录，is_zipfile 为 False，据此跳过损坏文件；
    不用 testzip 逐字节校验，因为单个 checkpoint 约 2GB，在 NFS 上太慢。
    """
    cands = [out_dir / "checkpoint.pth"] + sorted(out_dir.glob("checkpoint[0-9]*.pth"), reverse=True)
    return next((p for p in cands if p.is_file() and zipfile.is_zipfile(p)), None)


def with_resume(cmd: str, ckpt: Path) -> str:
    """给命令加上 `--resume <ckpt>`；命令里已有 --resume 则替换（脚本把额外参数放在最后，argparse 以后者为准）。"""
    arg = f"--resume {shlex.quote(str(ckpt))}"
    return RESUME_ARG_RE.sub(lambda _: arg, cmd, count=1) if RESUME_ARG_RE.search(cmd) else f"{cmd} {arg}"


class Supervisor:
    def __init__(self, task_file: Path, state_dir: Path, log_dir: Path, tmux: Tmux,
                 max_retries: int = 3, only: Optional[List[str]] = None,
                 auto_resume: bool = True, echo=lambda s: print(s, flush=True)):
        self.task_file, self.state_dir, self.log_dir = Path(task_file), Path(state_dir), Path(log_dir)
        self.tmux, self.max_retries, self.only, self.echo = tmux, max_retries, only, echo
        self.auto_resume = auto_resume
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 状态与事件 ----------
    def _state_path(self, name: str) -> Path:
        return self.state_dir / f"{safe(name)}.json"

    def load_state(self, name: str) -> dict:
        try:
            return json.loads(self._state_path(name).read_text())
        except FileNotFoundError:
            return {"status": "idle", "run_id": None, "attempts": 0}

    def _save(self, name: str, st: dict) -> None:
        tmp = self._state_path(name).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
        tmp.replace(self._state_path(name))

    def event(self, name: str, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} [{name}] {msg}"
        self.echo(line)
        with open(self.state_dir / "events.log", "a") as f:
            f.write(line + "\n")

    @contextmanager
    def _window_lock(self, name: str):
        # 每个窗口一把锁：允许多个监督进程分别监督不同窗口，同一窗口不会被并发处理。
        with open(self.state_dir / f"{safe(name)}.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True

    # ---------- 对外接口 ----------
    def window_names(self) -> List[str]:
        names = list(load_windows(self.task_file))
        return [n for n in names if self.only is None or n in self.only]

    def exit_pending(self, name: str) -> bool:
        """当前任务的退出记录是否已出现（用于不等满周期就立即处理）。"""
        ef = self.load_state(name).get("exit_file")
        return bool(ef) and Path(ef).exists()

    def check(self, name: str) -> None:
        with self._window_lock(name) as ok:
            if ok:
                self._check(name)

    def resume(self, name: str) -> None:
        st = self.load_state(name)
        if st["status"] == "paused":
            st.update(status="idle", attempts=0, run_id=None, exit_file=None, reason=None)
            self._save(name, st)
            self.event(name, "已手动恢复")

    # ---------- 单窗口检查 ----------
    def _check(self, name: str) -> None:
        try:
            w = load_windows(self.task_file).get(name)
        except (OSError, ValueError) as e:
            self.event(name, f"异常：读取 {self.task_file} 失败：{e}")
            return
        if w is None:
            return
        st = self.load_state(name)
        st["last_check"] = datetime.now().isoformat(timespec="seconds")
        if st["status"] == "paused":
            self._save(name, st)
            return

        pid = self.tmux.pane_pid(name)
        if pid is None:
            if st["status"] != "no-session":
                self.event(name, "tmux 窗口不存在；等待其出现（不会自动创建）")
            st["status"] = "no-session"
            self._save(name, st)
            return
        if st["status"] == "no-session":
            st["status"] = "running" if st.get("run_id") else "idle"

        if st.get("run_id"):
            exit_file = Path(st["exit_file"])
            if exit_file.exists():
                # 退出记录已写入，但 tee 可能还没收尾；等 shell 空闲再动作，否则下一轮重试
                if not self.tmux.wait_idle(pid, 10):
                    self._save(name, st)
                    return
                rc = int(exit_file.read_text().strip() or 1)
                exit_file.unlink()
                self._on_exit(w, st, rc)
            elif self.tmux.busy(pid):
                st["status"] = "running"
                self._save(name, st)
            elif time.time() - st["started_at"] < START_GRACE:
                self._save(name, st)
            else:
                self._pause(w, st, "窗口已空闲但没有退出记录，疑似被人工中断或终端被关闭")
            return

        if self.tmux.busy(pid):
            st["status"] = "external"  # 窗口里有不是本工具启动的命令在跑，等它结束
            self._save(name, st)
            return
        self._start_next(w, st)

    def _pause(self, w: Window, st: dict, reason: str) -> None:
        st.update(status="paused", reason=reason, run_id=None, exit_file=None)
        self._save(w.name, st)
        self.event(w.name, f"暂停：{reason}（处理后执行 resume {w.name}）")

    def _on_exit(self, w: Window, st: dict, rc: int) -> None:
        cmd = st["command"]
        if rc == 0:
            self.event(w.name, f"完成：{cmd}")
            st.update(status="idle", run_id=None, exit_file=None, attempts=0)
            self._start_next(w, st)
        elif rc in INTERRUPT_RCS:
            self._pause(w, st, f"命令收到人工中断（退出码 {rc}）：{cmd}")
        else:
            self._retry(w, st, rc)

    def _retry(self, w: Window, st: dict, rc: int) -> None:
        cmd = st["command"]
        if st["attempts"] >= self.max_retries:
            return self._pause(w, st, f"重试 {self.max_retries} 次仍失败（退出码 {rc}）：{cmd}")
        if OOM_RE.search(self._log_tail(st["log"])):
            # 仅 CUDA OOM 才减小 batch_size；其他错误（读写错误等）保持原命令重试
            hits = list(BATCH_ARG_RE.finditer(cmd))
            if not hits or int(hits[-1].group(2)) <= 1:
                return self._pause(w, st, f"CUDA OOM，但命令中没有可再减小的 --batch_size N：{cmd}")
            m = hits[-1]  # 多次出现时 argparse 以最后一个为准
            old = int(m.group(2))
            new = max(1, old // 2)
            self.event(w.name, f"CUDA OOM：--batch_size {old} -> {new}")
            cmd = cmd[:m.start(2)] + str(new) + cmd[m.end(2):]
            st["command"] = cmd  # 后续重试在已减半的命令上继续减半
        else:
            self.event(w.name, f"命令失败（退出码 {rc}），原配置重试")
        st["attempts"] += 1
        self.event(w.name, f"第 {st['attempts']}/{self.max_retries} 次重试")
        self._launch(w, st, cmd, self._resume_cmd(w, st, cmd))

    def _resume_cmd(self, w: Window, st: dict, cmd: str) -> str:
        """重试时从上一次运行写出的最新 checkpoint 续训；找不到就原样重跑。

        run_cmd 只用于本次发送，st["command"] 仍保留不带自动 --resume 的原命令，
        这样下一次重试会重新从最新的 checkpoint 推导，而不是沿用过期的路径。
        """
        if not self.auto_resume:
            return cmd
        m = OUTPUT_DIR_RE.search(self._log_head(st["log"]))
        if not m:
            self.event(w.name, "日志里没有 output_dir，无法定位 checkpoint，原命令重试")
            return cmd
        out_dir = Path(w.workdir) / m.group(1)  # 绝对路径时 Path 会忽略前缀
        ckpt = find_checkpoint(out_dir)
        if ckpt is None:
            self.event(w.name, f"{out_dir} 中没有可用的 checkpoint，从头重跑")
            return cmd
        self.event(w.name, f"自动续训：--resume {ckpt}")
        return with_resume(cmd, ckpt)

    def _start_next(self, w: Window, st: dict) -> None:
        if not w.tasks:
            st["status"] = "idle"
            st["reason"] = "队列已空"
            self._save(w.name, st)
            return
        line = w.tasks[0]
        cmd = line
        if subprocess.run(["bash", "-n", "-c", cmd], capture_output=True).returncode != 0:
            return self._pause(w, st, f"队列首条命令有 shell 语法错误，未执行：{cmd}")
        if not move_first_task_to_worked(self.task_file, w.name, line):
            self.event(w.name, "task 文件在读取后被修改，下一轮再试")
            self._save(w.name, st)
            return
        st["attempts"] = 0
        st["reason"] = None
        self._launch(w, st, cmd)

    def _launch(self, w: Window, st: dict, cmd: str, run_cmd: Optional[str] = None) -> None:
        run_cmd = run_cmd or cmd
        run_id = str(time.time_ns())
        log = self.log_dir / f"{safe(w.name)}-{run_id}.log"
        exit_file = self.state_dir / f"{safe(w.name)}.{run_id}.exit"
        # 子 shell + `bash -c`：任务里的 exit/cd 不会影响窗口 shell；退出码先写临时文件再 mv，保证读到的是完整内容。
        wrapped = (f"( cd {shlex.quote(w.workdir)} && bash -c {shlex.quote(run_cmd)}; rc=$?; "
                   f"echo $rc > {shlex.quote(str(exit_file))}.tmp && "
                   f"mv {shlex.quote(str(exit_file))}.tmp {shlex.quote(str(exit_file))}; exit $rc ) "
                   f"2>&1 | tee -a {shlex.quote(str(log))}")
        st.update(status="running", run_id=run_id, command=cmd, log=str(log),
                  exit_file=str(exit_file), started_at=time.time())
        self._save(w.name, st)  # 先落盘再发送，进程崩溃重启后仍能接上这次运行
        self.event(w.name, f"启动：{run_cmd}")
        self.tmux.send(w.name, wrapped)

    @staticmethod
    def _log_head(path: str) -> str:
        try:
            with open(path, "rb") as f:
                return f.read(LOG_TAIL_BYTES).decode("utf-8", "replace")
        except OSError:
            return ""

    @staticmethod
    def _log_tail(path: str) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - LOG_TAIL_BYTES))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    # ---------- 主循环 ----------
    def run_forever(self, interval: float, poll: float = 2.0) -> None:
        """每 interval 秒全量检查一次；其间每 poll 秒看一眼有没有任务刚结束，有就立即处理。"""
        next_full = 0.0
        while True:
            full = time.time() >= next_full
            if full:
                next_full = time.time() + interval
            try:
                for n in self.window_names():
                    if full or self.exit_pending(n):
                        self.check(n)
            except (OSError, ValueError) as e:
                self.echo(f"异常：{e}")
            time.sleep(poll)
