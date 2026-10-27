"""核心监督逻辑：一个 Supervisor 只监督一个 tmux 窗口，反复 检查 -> (重试 | 移入 failed | 启动队列下一条 | 结束监控)。

任务通过 send-keys 发到窗口的交互 shell 里执行，包装命令在结束时把退出码写入
`<state_dir>/<窗口>.<run_id>.exit`，监督器据此判断成功/失败，不依赖解析屏幕输出。
"""
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .tasks import Window, load_windows, move_task, parse_task
from .tmuxctl import Tmux, TmuxError

# 只匹配 CUDA 显存不足；"illegal memory access" 之类不是 OOM，减小 batch 没用。
OOM_RE = re.compile(r"CUDA out of memory|CUDA error:.*out of memory|CUBLAS_STATUS_ALLOC_FAILED"
                    r"|CUDNN_STATUS_ALLOC_FAILED", re.I)
# 命令行里的 batch_size 参数（支持 `--batch_size 4` 与 `--batch_size=4`），OOM 重试时改写它
BATCH_ARG_RE = re.compile(r"(--batch_size[ =])(\d+)")
# 训练程序启动时会 print(args)，其中的 output_dir 即本次运行实际写 checkpoint 的目录（相对任务的 workdir）
OUTPUT_DIR_RE = re.compile(r"output_dir='([^']+)'")
RESUME_ARG_RE = re.compile(r"--resume(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|\S+)")
# 130/143：Ctrl-C / SIGTERM，按"手动中断"处理：任务移入 failed list，继续下一条。
# 120：Python 退出时刷新 stdout 失败，Ctrl-C 经 tee 时常见，但磁盘写满也会出现；
# 只有日志末尾带 KeyboardInterrupt 才算手动中断，否则当普通错误重试。
INTERRUPT_RCS = (130, 143)
INTERRUPT_LOG_RE = re.compile(r"KeyboardInterrupt")
START_GRACE = 15  # 发出命令后，等多久才允许判定"窗口空闲却没有退出记录"（shell 启动子进程需要时间）
LOG_TAIL_BYTES = 200_000  # 判断错误类型时只读日志末尾；找 output_dir 时读日志开头同样大小


class MonitorStop(Exception):
    """无法再安全地自动处理，需要结束监控（任务保持在 task-list 中）。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AlreadyRunning(Exception):
    def __init__(self, pid: Optional[int]):
        super().__init__(f"a monitor is already running (pid {pid})")
        self.pid = pid


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


# ---------- 每个窗口只允许一个监控进程：文件锁（进程结束自动释放）+ 锁文件里记录 pid 和窗口名 ----------
def acquire_lock(state_dir: Path, name: str):
    state_dir.mkdir(parents=True, exist_ok=True)
    f = open(state_dir / f"{safe(name)}.lock", "a+")  # 先不截断：别的进程可能正持有并写着自己的 pid
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        first = f.readline().strip()
        f.close()
        raise AlreadyRunning(int(first) if first.isdigit() else None)
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()}\n{name}\n")
    f.flush()
    return f  # 调用方持有它直到监控结束


def running_monitors(state_dir: Path) -> Dict[str, int]:
    """{窗口名: pid}，只含当前确实持有锁（即正在运行）的监控。"""
    out: Dict[str, int] = {}
    for p in Path(state_dir).glob("*.lock"):
        with open(p, "r+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lines = f.read().split("\n")
                if lines and lines[0].isdigit() and len(lines) > 1:
                    out[lines[1]] = int(lines[0])
    return out


class Supervisor:
    def __init__(self, task_file: Path, state_dir: Path, log_dir: Path, tmux: Tmux, name: str,
                 max_retries: int = 3, retry_delay: float = 10.0, auto_resume: bool = True,
                 start_grace: float = START_GRACE, echo=lambda s: print(s, flush=True)):
        self.task_file, self.state_dir, self.log_dir = Path(task_file), Path(state_dir), Path(log_dir)
        self.tmux, self.name = tmux, name
        self.max_retries, self.retry_delay, self.auto_resume = max_retries, retry_delay, auto_resume
        self.start_grace, self.echo = start_grace, echo
        self._stop = False
        self.finished = False  # 队列跑空且没有任务在运行：监控完成使命，run_forever 据此退出
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 状态与事件 ----------
    @property
    def _state_path(self) -> Path:
        return self.state_dir / f"{safe(self.name)}.json"

    def load_state(self) -> dict:
        try:
            return json.loads(self._state_path.read_text())
        except FileNotFoundError:
            return {"status": "idle", "run_id": None, "attempts": 0}

    def _save(self, st: dict) -> None:
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
        tmp.replace(self._state_path)

    def reset(self) -> None:
        """清掉状态文件和遗留的退出记录，下次 run 是全新的监控（stop / 队列跑空 / 自动结束时调用；Ctrl-C 不调用）。"""
        self._state_path.unlink(missing_ok=True)
        for p in self.state_dir.glob(f"{safe(self.name)}.*.exit*"):
            p.unlink(missing_ok=True)

    def event(self, msg: str, extra: List[str] = ()) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} [{self.name}] {msg}"
        text = "\n".join([line] + [f"    {l}" for l in extra])
        self.echo(text)
        with open(self.state_dir / "events.log", "a") as f:
            f.write(text + "\n")

    def request_stop(self) -> None:
        self._stop = True

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while not self._stop and time.time() < end:
            time.sleep(min(0.2, max(0.0, end - time.time())))

    # ---------- 单次检查 ----------
    def needs_attention(self) -> bool:
        """不等满周期就该立即检查吗：当前任务的退出记录已出现，或任务应在运行但窗口已空闲（没有退出记录的中断）。"""
        st = self.load_state()
        if not st.get("run_id"):
            return False
        if Path(st["exit_file"]).exists():
            return True
        try:
            pid = self.tmux.pane_pid(self.name)
            return pid is not None and not self.tmux.busy(pid)
        except TmuxError as e:
            raise MonitorStop(f"tmux did not respond: {e}")

    def _window(self) -> Window:
        w = load_windows(self.task_file).get(self.name)
        if w is None:
            raise MonitorStop(f"the task file has no queue for window {self.name}")
        return w

    def check(self) -> None:
        """检查一次；遇到无法自动处理的情况抛出 MonitorStop。"""
        try:
            self._check()
        except TmuxError as e:  # tmux 卡住：不再继续，免得监控无声冻结
            raise MonitorStop(f"tmux did not respond: {e}")

    def _check(self) -> None:
        try:
            w = self._window()
        except (OSError, ValueError) as e:
            self.event(f"error: failed to read {self.task_file}: {e}")
            return
        pid = self.tmux.pane_pid(self.name)
        if pid is None:
            raise MonitorStop(f"tmux window {self.name} does not exist or was closed")
        st = self.load_state()
        st["last_check"] = datetime.now().isoformat(timespec="seconds")

        if st.get("run_id"):
            exit_file = Path(st["exit_file"])
            if exit_file.exists():
                # 退出记录已写入，但 tee 可能还没收尾；等 shell 空闲再动作，否则下一轮再处理
                if not self.tmux.wait_idle(pid, 10):
                    self._save(st)
                    return
                rc = int(exit_file.read_text().strip() or 1)
                exit_file.unlink()
                self._on_exit(w, st, rc)
            elif self.tmux.busy(pid):
                st["status"] = "running"
                self._save(st)
            elif time.time() - st["started_at"] < self.start_grace:
                self._save(st)
            else:
                # 没有退出记录：进程被杀或终端被关，按手动中断处理
                self._fail(st, "manually interrupted (no exit record: process killed or terminal closed)", log_lines=True)
            return

        if self.tmux.busy(pid):
            if st["status"] != "external":
                self.event("a command not started by this tool is running in the window; waiting for it to finish (it will not be retried)")
            st["status"] = "external"
            self._save(st)
            return
        self._start_next(w, st)

    def _clear_run(self, st: dict) -> None:
        st.update(status="idle", run_id=None, exit_file=None, attempts=0)
        self._save(st)

    def _on_exit(self, w: Window, st: dict, rc: int) -> None:
        cmd = st["command"]
        if rc == 0:
            self.event(f"done: {cmd}")
            if not move_task(self.task_file, self.name, st["raw"], "worked", self._done_note(st)):
                self.event("warning: this command is no longer in task-list; not recorded in worked list")
            self._clear_run(st)
            self._start_next(self._window(), st)
        elif rc in INTERRUPT_RCS or (rc == 120 and INTERRUPT_LOG_RE.search(self._log_tail(st["log"]))):
            self._fail(st, f"manually interrupted (exit code {rc})")
        else:
            self._retry(w, st, rc)

    def _done_note(self, st: dict) -> str:
        """worked list 条目的注释：本次成功运行的日志位置，以及日志里 output_dir 对应的输出目录（若有）。"""
        note = f"log={st['log']}"
        m = OUTPUT_DIR_RE.search(self._log_head(st["log"]))
        if m:
            note += f" output_dir={os.path.normpath(os.path.join(st['workdir'], m.group(1)))}"
        return note

    def _fail(self, st: dict, note: str, log_lines: bool = False) -> None:
        """把当前任务移入 failed list，把原因打印到终端，窗口继续下一条。"""
        extra = self._last_lines(st.get("log"), 3) if log_lines else []
        if move_task(self.task_file, self.name, st["raw"], "failed", note):
            self.event(f"FAILED: {st['raw']} -> moved to failed list: {note}", extra)
        else:
            self.event(f"FAILED: {st['raw']}: {note} (command no longer in task-list; not written to failed list)", extra)
        self._clear_run(st)
        self._start_next(self._window(), st)

    def _retry(self, w: Window, st: dict, rc: int) -> None:
        cmd = st["command"]
        if st["attempts"] >= self.max_retries:
            return self._fail(st, f"exceeded max_retries={self.max_retries} (exit code {rc})", log_lines=True)
        if OOM_RE.search(self._log_tail(st["log"])):
            # 仅 CUDA OOM 才减小 batch_size；其他错误（读写错误等）保持原命令重试
            hits = list(BATCH_ARG_RE.finditer(cmd))
            if not hits or int(hits[-1].group(2)) <= 1:
                return self._fail(st, "CUDA OOM and the command has no reducible --batch_size N", log_lines=True)
            m = hits[-1]  # 多次出现时 argparse 以最后一个为准
            old = int(m.group(2))
            new = max(1, old // 2)
            self.event(f"CUDA OOM: --batch_size {old} -> {new}")
            cmd = cmd[:m.start(2)] + str(new) + cmd[m.end(2):]
        else:
            self.event(f"command failed (exit code {rc}); retrying unchanged")
        st["attempts"] += 1
        st["command"] = cmd  # 后续重试在已减半的命令上继续减半
        self.event(f"retry {st['attempts']}/{self.max_retries}, starting in {self.retry_delay:g}s")
        self._sleep(self.retry_delay)
        if self._stop:  # 等待期间收到 stop：不再启动，任务仍在 task-list
            return self._clear_run(st)
        self._launch(w, st, cmd, self._resume_cmd(st, cmd))

    def _resume_cmd(self, st: dict, cmd: str) -> str:
        """重试时从上一次运行写出的最新 checkpoint 续训；找不到就原样重跑。

        run_cmd 只用于本次发送，st["command"] 仍保留不带自动 --resume 的原命令，
        这样下一次重试会重新从最新的 checkpoint 推导，而不是沿用过期的路径。
        """
        if not self.auto_resume:
            return cmd
        m = OUTPUT_DIR_RE.search(self._log_head(st["log"]))
        if not m:
            self.event("no output_dir in the log; cannot locate a checkpoint; retrying unchanged")
            return cmd
        out_dir = Path(st["workdir"]) / m.group(1)  # 绝对路径时 Path 会忽略前缀
        ckpt = find_checkpoint(out_dir)
        if ckpt is None:
            self.event(f"no usable checkpoint in {out_dir}; restarting from scratch")
            return cmd
        self.event(f"auto-resume: --resume {ckpt}")
        return with_resume(cmd, ckpt)

    def _start_next(self, w: Window, st: dict) -> None:
        while True:
            if not w.tasks:
                st.update(status="idle", reason="queue empty")
                self._save(st)
                self.finished = True
                self.event(f"all tasks finished (worked list: {len(w.worked)}, failed list: {len(w.failed)}); monitor exits")
                return
            raw = w.tasks[0]
            cmd, workdir = parse_task(raw, w.workdir)
            if subprocess.run(["bash", "-n", "-c", cmd], capture_output=True).returncode == 0:
                break
            if not move_task(self.task_file, self.name, raw, "failed", "shell syntax error"):
                self.event("the task file changed after it was read; will retry next round")
                return
            self.event(f"FAILED: {raw} -> moved to failed list: shell syntax error")
            w = self._window()
        # 命令留在 task-list 首行，成功后才移走（见 _on_exit）
        st.update(raw=raw, attempts=0, workdir=workdir, reason=None)
        self._launch(w, st, cmd)

    def _launch(self, w: Window, st: dict, cmd: str, run_cmd: Optional[str] = None) -> None:
        run_cmd = run_cmd or cmd
        run_id = str(time.time_ns())
        log = self.log_dir / f"{safe(self.name)}-{run_id}.log"
        exit_file = self.state_dir / f"{safe(self.name)}.{run_id}.exit"
        # 子 shell + `bash -c`：任务里的 exit/cd 不会影响窗口 shell；退出码先写临时文件再 mv，保证读到的是完整内容。
        wrapped = (f"( cd {shlex.quote(st['workdir'])} && bash -c {shlex.quote(run_cmd)}; rc=$?; "
                   f"echo $rc > {shlex.quote(str(exit_file))}.tmp && "
                   f"mv {shlex.quote(str(exit_file))}.tmp {shlex.quote(str(exit_file))}; exit $rc ) "
                   f"2>&1 | tee -a {shlex.quote(str(log))}")
        st.update(status="running", run_id=run_id, command=cmd, log=str(log),
                  exit_file=str(exit_file), started_at=time.time())
        self._save(st)  # 先落盘再发送，进程崩溃重启后仍能接上这次运行
        self.event(f"start: {run_cmd}")
        self.tmux.send(self.name, wrapped)

    # ---------- 日志读取 ----------
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

    def _last_lines(self, path: Optional[str], n: int) -> List[str]:
        lines = [l.rstrip()[:200] for l in self._log_tail(path or "").splitlines() if l.strip()]
        return lines[-n:]

    # ---------- 主循环 ----------
    def run_forever(self, interval: float, poll: float = 2.0) -> int:
        """每 interval 秒全量检查一次；其间每 poll 秒看一眼有没有任务刚结束，有就立即处理。

        返回退出码：0 = 队列跑空 / 被 stop / Ctrl-C（仅 Ctrl-C 保留状态）正常结束；1 = 因无法自动处理的情况自动结束。
        """
        next_full = 0.0
        try:
            while not self._stop:
                full = time.time() >= next_full
                if full:
                    next_full = time.time() + interval
                if full or self.needs_attention():
                    try:
                        self.check()
                    except (OSError, ValueError) as e:
                        self.event(f"error: {e}")
                    if self.finished:
                        self.reset()
                        return 0
                self._sleep(poll)
        except MonitorStop as e:
            self.event(f"monitor ended: {e.reason}")
            self.reset()
            return 1
        except KeyboardInterrupt:
            # Ctrl-C 只是暂离：保留状态文件，重新 run 时接着监控正在跑的任务（不会当成外部命令、也不会重复启动）
            self.event("received Ctrl-C; monitor ended, state kept: the next `run` resumes monitoring (tasks in tmux are not affected)")
            return 0
        self.event("received stop command; monitor ended (tasks in tmux are not affected)")
        self.reset()  # stop 才是"结束这次监控"：下次 run 是全新的监控
        return 0
