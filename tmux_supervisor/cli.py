import argparse
import os
import signal
import sys
import time
from pathlib import Path

from .supervisor import AlreadyRunning, MonitorStop, Supervisor, acquire_lock, clear_state, running_monitors, safe
from .tasks import load_windows
from .tmuxctl import Tmux

ROOT = Path(__file__).resolve().parent.parent  # 工具主文件夹：task.md、state/、logs/ 都放这里


def build_parser() -> argparse.ArgumentParser:
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--task-file", type=Path, default=ROOT / "task.md", help="queue file (default: task.md in the tool folder)")
    paths.add_argument("--state-dir", type=Path, default=ROOT / "state")
    paths.add_argument("--log-dir", type=Path, default=ROOT / "logs")

    opts = argparse.ArgumentParser(add_help=False, parents=[paths])
    opts.add_argument("--max-retries", type=int, default=3, help="max retries per command (default 3)")
    opts.add_argument("--retry-delay", type=float, default=10, help="seconds to wait between retries (default 10)")
    opts.add_argument("--no-auto-resume", action="store_true",
                      help="do not add --resume on retry (default: resume from the checkpoint the failed run left behind, if any)")
    opts.add_argument("--socket", help="tmux -L socket name (default: the default server)")

    p = argparse.ArgumentParser(prog="tmuxsup", description="Supervise training tasks in one tmux window")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", parents=[opts], help="supervise one window continuously")
    run.add_argument("window", help="tmux window (session name, or session:window)")
    run.add_argument("--interval", type=float, default=1800, help="full check interval in seconds (default 1800); a finished task is handled immediately")
    run.add_argument("--poll", type=float, default=2, help="polling interval in seconds for detecting a finished task (default 2)")
    once = sub.add_parser("once", parents=[opts], help="do one check, then exit")
    once.add_argument("window")
    status = sub.add_parser("status", parents=[opts], help="show window state and queue sizes")  # 与 run/once 接受同一组选项，便于复用同一串参数
    status.add_argument("window")
    stop = sub.add_parser("stop", help="end monitor process(es); tasks in tmux are not affected")
    stop.add_argument("window", nargs="?", help="window whose monitor to end")
    stop.add_argument("--all", action="store_true", help="end all monitors")
    stop.add_argument("--state-dir", type=Path, default=ROOT / "state")
    return p


def stop_monitors(state_dir: Path, window, all_: bool) -> int:
    mons = running_monitors(state_dir)
    targets = mons if all_ else ({window: mons[window]} if window in mons else {})
    rc = 0
    for name, pid in targets.items():
        os.kill(pid, signal.SIGTERM)
        end = time.time() + 15
        while time.time() < end and name in running_monitors(state_dir):
            time.sleep(0.2)
        if name in running_monitors(state_dir):
            print(f"monitor {name} (pid {pid}) did not exit within 15s", file=sys.stderr)
            rc = 1
        else:
            print(f"stopped monitor {name} (pid {pid})")  # 状态由监控进程退出时自己清掉
    # stop 的语义是"下次 run 全新开始"。监控早已退出（Ctrl-C、被杀、终端关闭）时没有进程替它清状态，这里补上，
    # 否则下次 run 会接着旧状态把"已不在跑的任务"当成被中断。仍在运行的监控（没退出的）不碰。
    alive = running_monitors(state_dir)
    stems = ([p.stem for p in Path(state_dir).glob("*.json")] if all_ else [safe(window)])
    cleared = [st for st in stems if st not in {safe(n) for n in alive} and clear_state(state_dir, st)]
    head = "No monitor is running" + ("" if all_ else f": {window}")
    if cleared:
        print(("" if targets else f"{head}; ") + f"cleared saved state of: {', '.join(cleared)}", file=sys.stderr)
        return rc
    if not targets:
        print(head, file=sys.stderr)
        return 0 if all_ else 1
    return rc


def show_status(a) -> int:
    wins = load_windows(a.task_file)
    w = wins[a.window]
    sup = Supervisor(a.task_file, a.state_dir, a.log_dir, Tmux(), a.window)
    st = sup.load_state()
    pid = running_monitors(a.state_dir).get(a.window)
    print(f"[{a.window}] monitor: {'running, pid ' + str(pid) if pid else 'not running'}  state: {st['status']}"
          f"  last check: {st.get('last_check', '-')}")
    print(f"    current: {st.get('command') if st.get('run_id') else '-'}  retries: {st.get('attempts', 0)}")
    if not pid and st.get("run_id"):  # 监控已结束，状态文件停在结束前的样子，不代表任务现在的情况
        print("    (the monitor is not running; this is the last recorded state, not live)")
    print(f"    task-list: {len(w.tasks)}  worked: {len(w.worked)}  failed: {len(w.failed)}")
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "stop":
        if bool(a.window) == a.all:
            print("stop needs a window name or --all (exactly one of them)", file=sys.stderr)
            return 2
        return stop_monitors(a.state_dir, a.window, a.all)

    try:
        windows = load_windows(a.task_file)
    except (OSError, ValueError) as e:
        print(f"failed to read {a.task_file}: {e}", file=sys.stderr)
        return 2
    if a.window not in windows:
        print(f"the task file has no window {a.window} (available: {list(windows)})", file=sys.stderr)
        return 2
    if a.cmd == "status":
        return show_status(a)

    try:
        lock = acquire_lock(a.state_dir, a.window)  # 一个窗口只允许一个监控；持有到进程结束
    except AlreadyRunning as e:
        print(f"A monitor is already running for window {a.window} (pid {e.pid}); stop it first with `tmuxsup stop {a.window}`", file=sys.stderr)
        return 1
    sup = Supervisor(a.task_file, a.state_dir, a.log_dir, Tmux(a.socket), a.window, a.max_retries,
                     a.retry_delay, auto_resume=not a.no_auto_resume)
    if a.cmd == "once":
        try:
            sup.check()
        except MonitorStop as e:
            sup.event(f"monitor ended: {e.reason}")
            return 1
        return 0

    signal.signal(signal.SIGTERM, lambda *_: sup.request_stop())  # `tmuxsup stop` 发来的
    print(f"Monitoring window {a.window}: interval {a.interval:g}s, max retries {a.max_retries}, queue {a.task_file}", flush=True)
    code = sup.run_forever(a.interval, a.poll)
    lock.close()
    return code
