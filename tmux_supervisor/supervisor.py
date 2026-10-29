"""核心监督逻辑：一个 Supervisor 只监督一个 tmux 窗口，反复 检查 -> (重试 | 移入 failed | 启动队列下一条 | 结束监控)。

任务通过 send-keys 发到窗口的交互 shell 里执行，包装命令在结束时把退出码写入
`<state_dir>/<窗口>.<run_id>.exit`，监督器据此判断成功/失败，不依赖解析屏幕输出。
"""
import fcntl
import json
import os
import pickle
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

# 只匹配 GPU 显存不足；"illegal memory access"、CPU 内存不足（DefaultCPUAllocator / Killed）不是 OOM，减小 batch 没用。
# 各行依次是：torch.cuda.OutOfMemoryError / torch.OutOfMemoryError；同一行里 cuda/hip 与 out of memory 同现
# （CUDA out of memory、CUDA error: out of memory、CUDA driver error、NCCL 转述的 cuda 错误、HIP out of memory）；
# 错误码标识符；PyTorch 的 "Tried to allocate ... (GPU 0; ...)"；cuBLAS/cuDNN/cuSOLVER 等的 *_STATUS_ALLOC_FAILED。
OOM_RE = re.compile(r"OutOfMemoryError"
                    r"|(?:\bcuda|\bhip)[^\n]*out of memory"
                    r"|CUDA_ERROR_OUT_OF_MEMORY|(?:cuda|hip)Error(?:MemoryAllocation|OutOfMemory)"
                    r"|Tried to allocate [\d.]+ \w+ \(GPU \d"
                    r"|\bCU[A-Z]+_STATUS_ALLOC_FAILED", re.I)
# 端口被占用：torch 的 TCPStore 报 EADDRINUSE / "address already in use"（errno 98）。重试时换一个 MASTER_PORT
PORT_BUSY_RE = re.compile(r"EADDRINUSE|address already in use|errno[ :=]*98\b", re.I)
# 报错行里的端口号（`port: 29501,` 或 `[::]:29500 (errno: 98 ...`），用来判断换端口有没有生效
PORT_NUM_RE = re.compile(r"\bport:?\s*(\d+)|:(\d{4,5})\b", re.I)
# 由运行命令的那台机器自己向系统要一个空闲端口（窗口可能是 srun 进计算节点的嵌套 shell，监控本机探测的结果不一定适用）
FREE_PORT = "$(python3 -c 'import socket; s = socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])')"
MASTER_PORT_ARG_RE = re.compile(r"\bMASTER_PORT=\d+")
# 命令行里的 batch_size 参数（支持 `--batch_size 4` 与 `--batch_size=4`），OOM 重试时改写它
BATCH_ARG_RE = re.compile(r"(--batch_size[ =])(\d+)")
# 训练程序启动时 print(args) 里的 batch_size=N，是本次运行实际生效的值（batch_size 写在脚本里时命令行上看不到）
BATCH_LOG_RE = re.compile(r"\bbatch_size=(\d+)")
# accum_steps（梯度累积步数）的读法同 batch_size：先看命令行，再看日志 print(args) 里的 accum_steps=N
ACCUM_ARG_RE = re.compile(r"(--accum_steps[ =])(\d+)")
ACCUM_LOG_RE = re.compile(r"\baccum_steps=(\d+)")
# 训练程序启动时会 print(args)，其中的 output_dir 即本次运行实际写 checkpoint 的目录（相对任务的 workdir）
OUTPUT_DIR_RE = re.compile(r"output_dir='([^']+)'")
RESUME_ARG_RE = re.compile(r"--resume(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|\S+)")
# 130/143：Ctrl-C / SIGTERM，按"手动中断"处理：任务移入 failed list，继续下一条。
# torchrun 等程序捕获 SIGINT 后会以 1 退出，所以退出码非 0 时还看日志：带 KeyboardInterrupt，
# 或 torchrun 的 "got signal: 2/15"，也算手动中断。其余错误（含 120）当普通错误重试。
INTERRUPT_RCS = (130, 143)
INTERRUPT_LOG_RE = re.compile(r"KeyboardInterrupt|SignalException: .*got signal: (?:2|15)\b")
INTERRUPT_WAIT = 30  # 判为手动中断后，等多久才移入 failed list 并启动下一条（这段时间内 stop 可保住队列）
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


NUMBERED_CKPT_RE = re.compile(r"checkpoint(\d+)\.pth")  # 训练程序写作 checkpoint{epoch:04}.pth，数字即 epoch


class _Stub:
    """读 checkpoint 的 epoch 时代替 pickle 里的所有类和函数：只造空壳，不执行 pickle 携带的任何代码。"""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass


class _EpochUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Stub

    def persistent_load(self, pid):  # torch.save 把张量存储放在 zip 的其他条目里，这里不读
        return None


def read_epoch(path: Path) -> Optional[int]:
    """读出 torch.save 写的 checkpoint 里的 'epoch'；只解析 zip 里很小的 data.pkl，不加载约 2GB 的张量，也不依赖 torch。读不出返回 None。"""
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n == "data.pkl" or n.endswith("/data.pkl"))
            with z.open(name) as f:
                epoch = _EpochUnpickler(f).load()["epoch"]
        return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None
    except Exception:
        return None


def find_checkpoint(out_dir: Path) -> Optional[Path]:
    """返回目录里最新的完好 checkpoint：checkpoint.pth（每个 epoch 覆盖写）和 checkpointNNNN.pth 中 epoch 最大者。

    不再无条件优先 checkpoint.pth：它可能是旧运行遗留的，比编号文件旧。epoch 优先取文件内容里的 'epoch'，
    读不出的编号文件取文件名里的数字（按数值而不是字符串比较）；epoch 相同取修改时间新的。
    只要有候选读不出 epoch（checkpoint.pth 内容无法解析），所有候选就都退回按修改时间比较。

    torch.save 写的是 zip，被中途打断（如磁盘写满）的文件缺少文件尾的目录，is_zipfile 为 False，据此跳过损坏文件；
    不用 testzip 逐字节校验，因为单个 checkpoint 约 2GB，在 NFS 上太慢。
    """
    cands = [p for p in [out_dir / "checkpoint.pth", *out_dir.glob("checkpoint[0-9]*.pth")]
             if (p.name == "checkpoint.pth" or NUMBERED_CKPT_RE.fullmatch(p.name)) and p.is_file() and zipfile.is_zipfile(p)]
    if not cands:
        return None
    info = []  # (路径, epoch 或 None, 修改时间)
    for p in cands:
        epoch = read_epoch(p)
        m = NUMBERED_CKPT_RE.fullmatch(p.name)
        if epoch is None and m:
            epoch = int(m.group(1))
        info.append((p, epoch, p.stat().st_mtime))
    if all(e is not None for _, e, _ in info):
        return max(info, key=lambda t: (t[1], t[2]))[0]
    return max(info, key=lambda t: (t[2], -1 if t[1] is None else t[1]))[0]


def with_arg(cmd: str, arg_re: "re.Pattern", flag: str, value: int) -> str:
    """把命令里的 `flag N` 改成 value（多次出现改最后一个，argparse 以它为准）；命令里没有就追加到末尾。"""
    hits = list(arg_re.finditer(cmd))
    if hits:
        return cmd[:hits[-1].start(2)] + str(value) + cmd[hits[-1].end(2):]
    return f"{cmd} {flag} {value}"


def with_free_port(cmd: str) -> str:
    """让命令每次运行都取一个新的空闲 MASTER_PORT：命令里已有 `MASTER_PORT=数字` 就原位改写，否则在最前面 export。
    已经改过的命令原样返回（$(...) 每次运行都会重新取一个端口）。"""
    if FREE_PORT in cmd:
        return cmd
    if MASTER_PORT_ARG_RE.search(cmd):
        return MASTER_PORT_ARG_RE.sub(lambda m: f"MASTER_PORT={FREE_PORT}", cmd)
    return f"export MASTER_PORT={FREE_PORT}; {cmd}"


def busy_port(tail: str) -> Optional[int]:
    """日志末尾最后一条端口被占用的报错里的端口号；没有端口号返回 None。"""
    lines = [l for l in tail.splitlines() if PORT_BUSY_RE.search(l)]
    m = PORT_NUM_RE.search(lines[-1]) if lines else None
    return int(m.group(1) or m.group(2)) if m else None


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


def clear_state(state_dir: Path, stem: str) -> bool:
    """删掉 stem（safe(窗口名)）对应的状态文件和遗留的退出记录；返回是否真的删了东西。"""
    removed = False
    for p in [Path(state_dir) / f"{stem}.json", *Path(state_dir).glob(f"{stem}.*.exit*")]:
        if p.exists():
            p.unlink(missing_ok=True)
            removed = True
    return removed


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
                 start_grace: float = START_GRACE, interrupt_wait: float = INTERRUPT_WAIT,
                 echo=lambda s: print(s, flush=True)):
        self.task_file, self.state_dir, self.log_dir = Path(task_file), Path(state_dir), Path(log_dir)
        self.tmux, self.name = tmux, name
        self.max_retries, self.retry_delay, self.auto_resume = max_retries, retry_delay, auto_resume
        self.start_grace, self.interrupt_wait, self.echo = start_grace, interrupt_wait, echo
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
        clear_state(self.state_dir, safe(self.name))

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
            return pid is not None and not self.tmux.busy(pid, self.name)
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
                if not self.tmux.wait_idle(pid, 10, self.name):
                    self._save(st)
                    return
                rc = int(exit_file.read_text().strip() or 1)
                exit_file.unlink()
                self._on_exit(w, st, rc)
            elif self.tmux.busy(pid, self.name):
                st["status"] = "running"
                self._save(st)
            elif time.time() - st["started_at"] < self.start_grace:
                self._save(st)
            elif exit_file.exists():
                # 上面查退出记录之后、判断空闲之前任务刚好结束：退出记录是先于进程退出写出的，现在再查一次就在了，下一轮按正常结束处理
                self._save(st)
            else:
                # 没有退出记录：进程被杀或终端被关，按手动中断处理
                self._interrupted(st, "manually interrupted (no exit record: process killed or terminal closed)", log_lines=True)
            return

        if w.tasks and st["status"] != "external" and self.tmux.nested(pid):
            # 嵌套 shell 里残留文字会让光标行不以提示符结尾，看起来像有程序在跑。要发新命令前先清一次行再判断（只判断这一次）：
            # 清掉后光标行回到提示符就是空闲；仍然忙说明真有程序在跑，照常等待。
            # 只在刚进入等待时做一次，不是每个周期都发；C-a C-k 不是信号，不会杀掉程序。
            self.tmux.clear_line(self.name)
            time.sleep(0.3)  # 等 shell 重画提示符
        if self.tmux.busy(pid, self.name):
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
        elif rc in INTERRUPT_RCS or INTERRUPT_LOG_RE.search(self._log_tail(st["log"])):
            self._interrupted(st, f"manually interrupted (exit code {rc})")
        else:
            self._retry(w, st, rc)

    def _done_note(self, st: dict) -> str:
        """worked list 条目的注释：本次成功运行的日志位置，以及日志里 output_dir 对应的输出目录（若有）。"""
        note = f"log={st['log']}"
        m = OUTPUT_DIR_RE.search(self._log_head(st["log"]))
        if m:
            note += f" output_dir={os.path.normpath(os.path.join(st['workdir'], m.group(1)))}"
        return note

    def _interrupted(self, st: dict, note: str, log_lines: bool = False) -> None:
        """手动中断：先等 interrupt_wait 秒再按失败处理。这段时间里用户可以 stop / Ctrl-C 监控来保住队列，
        否则被中断的任务会立刻进 failed list、下一条立刻开跑。"""
        self.event(f"{note}: {st['raw']}; waiting {self.interrupt_wait:g}s before moving it to failed list and starting the next command "
                   "(`stop` the monitor now to keep the queue as is)")
        self._sleep(self.interrupt_wait)
        if self._stop:  # 等待期间收到 stop：任务留在 task-list 首行，什么都不动
            return
        self._fail(st, note, log_lines)

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
        tail = self._log_tail(st["log"])
        if OOM_RE.search(tail):
            # 仅 CUDA OOM 才减小 batch_size；其他错误（读写错误等）保持原命令重试
            hits = list(BATCH_ARG_RE.finditer(cmd))
            m_log = BATCH_LOG_RE.search(self._log_head(st["log"]))
            log_bs = int(m_log.group(1)) if m_log else None  # 本次运行实际生效的 batch_size
            if hits:
                old = int(hits[-1].group(2))  # 多次出现时 argparse 以最后一个为准
                if log_bs is not None and log_bs != old:
                    # 命令行上的 --batch_size 没生效（脚本没把额外参数转发给训练程序）：继续减半只会空转
                    return self._fail(st, f"CUDA OOM, but the log shows batch_size={log_bs} while the command has "
                                          f"--batch_size {old}: the script does not apply the --batch_size override "
                                          f"(it must pass extra arguments through)", log_lines=True)
            else:
                old = log_bs  # batch_size 写在脚本里：以日志里的实际值为准，减半后追加 --batch_size 覆盖它
            if old is None or old <= 1:
                return self._fail(st, "CUDA OOM and no reducible --batch_size (not in the command or the log, or already 1)",
                                  log_lines=True)
            new = max(1, old // 2)
            # 梯度累积：batch 减半的同时把 accum_steps 加倍，逻辑 batch（world_size * batch_size * accum_steps）保持不变；
            # 奇数 batch 除不尽时 accum_steps 向上取整，逻辑 batch 略增大。命令行上没有就读日志，再没有就当作 1。
            acc_hits = ACCUM_ARG_RE.findall(cmd)
            m_acc = ACCUM_LOG_RE.search(self._log_head(st["log"]))
            old_acc = int(acc_hits[-1][1]) if acc_hits else int(m_acc.group(1)) if m_acc else 1
            new_acc = -(-old * old_acc // new)
            self.event(f"CUDA OOM: --batch_size {old} -> {new}, --accum_steps {old_acc} -> {new_acc}")
            # 命令里没有就追加到末尾：脚本把 "$@" 放在自己的参数之后，argparse 取最后一个
            cmd = with_arg(cmd, BATCH_ARG_RE, "--batch_size", new)
            cmd = with_arg(cmd, ACCUM_ARG_RE, "--accum_steps", new_acc)
        elif PORT_BUSY_RE.search(tail):
            port = busy_port(tail)
            if st.get("port_swapped") and port is not None and port == st.get("busy_port"):
                # 已经让命令改用空闲端口，日志里报的却还是同一个端口：脚本没读 MASTER_PORT（如写死了 --master_port），继续重试只会空转
                return self._fail(st, f"port {port} is in use again after switching MASTER_PORT: the script does not use the "
                                      f"MASTER_PORT environment variable", log_lines=True)
            st["busy_port"], st["port_swapped"] = port, True
            self.event(f"port in use{f' ({port})' if port else ''}: retrying with a free MASTER_PORT chosen at start")
            cmd = with_free_port(cmd)
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
        st.update(raw=raw, attempts=0, workdir=workdir, reason=None, busy_port=None, port_swapped=False)  # 端口记录按任务算，不带到下一条
        self._launch(w, st, cmd)

    def _launch(self, w: Window, st: dict, cmd: str, run_cmd: Optional[str] = None) -> None:
        run_cmd = run_cmd or cmd
        pid = self.tmux.pane_pid(self.name)
        if pid is None:
            raise MonitorStop(f"tmux window {self.name} does not exist or was closed")
        # 先清掉输入行上的残留字符，否则它们会和包装命令拼在一起导致语法错误；
        # 窗口已被外部程序占用（空闲检查之后才启动）时一个键都不发，免得 C-c 把它打断，任务留在 task-list 等它结束。
        if not self.tmux.clear_input(self.name, pid):
            self.event("a command not started by this tool is running in the window; will start the task after it finishes")
            return self._clear_run(st)
        run_id = str(time.time_ns())
        log = self.log_dir / f"{safe(self.name)}-{run_id}.log"
        exit_file = self.state_dir / f"{safe(self.name)}.{run_id}.exit"
        # 子 shell + `bash -c`：任务里的 exit/cd 不会影响窗口 shell；退出码先写临时文件再 mv，保证读到的是完整内容。
        # `tee -i` 忽略 SIGINT：Ctrl-C 会发给整个前台进程组，tee 若一并被杀，任务再往管道写就会失败，
        # Python 退出时 flush 报错得到退出码 120，真实退出码和收尾日志（如 KeyboardInterrupt、OOM）都丢失。
        wrapped = (f"( cd {shlex.quote(st['workdir'])} && bash -c {shlex.quote(run_cmd)}; rc=$?; "
                   f"echo $rc > {shlex.quote(str(exit_file))}.tmp && "
                   f"mv {shlex.quote(str(exit_file))}.tmp {shlex.quote(str(exit_file))}; exit $rc ) "
                   f"2>&1 | tee -i -a {shlex.quote(str(log))}")
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
