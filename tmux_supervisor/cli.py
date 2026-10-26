import argparse
import sys
from pathlib import Path

from .supervisor import Supervisor
from .tmuxctl import Tmux

ROOT = Path(__file__).resolve().parent.parent  # 工具主文件夹：task.md、state/、logs/ 都放这里


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task-file", type=Path, default=ROOT / "task.md", help="队列文件（默认主文件夹/task.md）")
    common.add_argument("--state-dir", type=Path, default=ROOT / "state")
    common.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    common.add_argument("--windows", nargs="+", metavar="NAME",
                        help="只监督这些 tmux 窗口（默认：task 文件中出现的全部）")
    common.add_argument("--max-retries", type=int, default=3, help="单条命令失败后的最大重试次数，默认 3")
    common.add_argument("--no-auto-resume", action="store_true",
                        help="重试时不自动加 --resume（默认：若上次运行留下了 checkpoint，则从它续训）")
    common.add_argument("--socket", help="tmux -L 的 socket 名（默认使用默认 server）")

    p = argparse.ArgumentParser(prog="tmuxsup", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", parents=[common], help="常驻监督")
    run.add_argument("--interval", type=float, default=1800, help="全量检查周期（秒），默认 1800；任务结束会立即处理")
    run.add_argument("--poll", type=float, default=2, help="检测任务结束的轮询间隔（秒）")
    sub.add_parser("once", parents=[common], help="检查一轮后退出")
    sub.add_parser("status", parents=[common], help="显示各窗口状态")
    res = sub.add_parser("resume", parents=[common], help="恢复被暂停的窗口")
    res.add_argument("names", nargs="+")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    sup = Supervisor(a.task_file, a.state_dir, a.log_dir, Tmux(a.socket), a.max_retries, a.windows,
                     auto_resume=not a.no_auto_resume)
    try:
        names = sup.window_names()
    except (OSError, ValueError) as e:
        print(f"读取 {a.task_file} 失败：{e}", file=sys.stderr)
        return 2
    unknown = [n for n in (a.windows or []) if n not in names]
    if unknown:
        print(f"task 文件中没有这些窗口：{unknown}", file=sys.stderr)
        return 2

    if a.cmd == "run":
        print(f"监督窗口 {names}，周期 {a.interval:g}s，队列 {a.task_file}", flush=True)
        try:
            sup.run_forever(a.interval, a.poll)
        except KeyboardInterrupt:
            return 0
    elif a.cmd == "once":
        for n in names:
            sup.check(n)
    elif a.cmd == "resume":
        for n in a.names:
            sup.resume(n)
    else:
        from .tasks import load_windows
        wins = load_windows(a.task_file)
        for n in names:
            st = sup.load_state(n)
            print(f"[{n}] {st['status']}  队列剩余 {len(wins[n].tasks)} 条  重试 {st.get('attempts', 0)}/{a.max_retries}"
                  f"\n    当前: {st.get('command')}\n    说明: {st.get('reason') or '-'}"
                  f"  上次检查: {st.get('last_check', '-')}")
    return 0
