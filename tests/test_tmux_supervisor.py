import os
import pickle
from unittest import mock
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from tmux_supervisor.supervisor import (OOM_RE, AlreadyRunning, MonitorStop, Supervisor, acquire_lock,
                                        find_checkpoint, running_monitors, with_resume)
from tmux_supervisor.tasks import load_windows, move_task, parse_task
from tmux_supervisor.tmuxctl import Tmux, TmuxError

ROOT = Path(__file__).resolve().parent.parent

TASKS = """# tmux: a
workdir: /tmp

## task-list
# 注释
bash x.sh

[workdir=sub] bash y.sh --batch_size 2
bash z.sh  # [failed old failure mark]

## worked list
old

## failed list
bad  # [failed reason]

# tmux: b
## task-list
bash z.sh
"""


class TaskFileTest(unittest.TestCase):
    def setUp(self):
        self.f = Path(tempfile.mkdtemp()) / "task.md"
        self.f.write_text(TASKS)

    def test_parse(self):
        w = load_windows(self.f)
        self.assertEqual(w["a"].tasks, ["bash x.sh", "[workdir=sub] bash y.sh --batch_size 2",
                                        "bash z.sh  # [failed old failure mark]"])
        self.assertEqual(w["a"].worked, ["old"])
        self.assertEqual(w["a"].failed, ["bad  # [failed reason]"])  # failed list 不会被当成任务
        self.assertEqual(w["a"].workdir, "/tmp")
        self.assertEqual(w["b"].tasks, ["bash z.sh"])

    def test_parse_task(self):
        self.assertEqual(parse_task("bash x.sh", "/w"), ("bash x.sh", "/w"))
        self.assertEqual(parse_task("[workdir=sub] bash y.sh", "/w"), ("bash y.sh", "/w/sub"))
        self.assertEqual(parse_task("[workdir=/abs] bash y.sh", "/w"), ("bash y.sh", "/abs"))
        # 之前失败过的条目被移回队列时，剥掉固定格式的失败标记，否则自动加的 --resume 会被当成注释吞掉
        self.assertEqual(parse_task("bash z.sh  # [failed exceeded max_retries=3]", "/w"), ("bash z.sh", "/w"))
        self.assertEqual(parse_task("bash z.sh  # [done log=/x/y.log output_dir=/o]", "/w"), ("bash z.sh", "/w"))
        self.assertEqual(parse_task("echo '#' # 普通注释", "/w")[0], "echo '#' # 普通注释")

    def test_move_to_worked(self):
        self.assertFalse(move_task(self.f, "a", "bash nope", "worked"))
        self.assertTrue(move_task(self.f, "a", "bash x.sh", "worked"))
        w = load_windows(self.f)
        self.assertEqual(w["a"].worked, ["old", "bash x.sh"])
        self.assertEqual(len(w["a"].tasks), 2)
        self.assertEqual(w["b"].tasks, ["bash z.sh"])  # 其他窗口不受影响

    def test_move_to_worked_with_note(self):
        self.assertTrue(move_task(self.f, "a", "bash x.sh", "worked", "log=/l.log"))
        self.assertEqual(load_windows(self.f)["a"].worked, ["old", "bash x.sh  # [done log=/l.log]"])

    def test_move_to_failed_with_note_creates_section(self):
        self.assertTrue(move_task(self.f, "b", "bash z.sh", "failed", "shell syntax error"))
        w = load_windows(self.f)
        self.assertEqual(w["b"].tasks, [])
        self.assertEqual(w["b"].failed, ["bash z.sh  # [failed shell syntax error]"])
        self.assertIn("## failed list", self.f.read_text())

    def test_move_not_first_line(self):
        # 用户在运行期间在前面插入了新行：按内容匹配，移走的是正在运行的那一条
        self.assertTrue(move_task(self.f, "a", "bash z.sh  # [failed old failure mark]", "worked"))
        self.assertEqual(load_windows(self.f)["a"].tasks[:2], ["bash x.sh", "[workdir=sub] bash y.sh --batch_size 2"])


def plain(lines):
    """去掉程序附加的 `# [done ...]` 注释，只比较命令本身。"""
    return [re.sub(r"  # \[done .*\]$", "", l) for l in lines]


def make_ckpt(path: Path, valid: bool = True, epoch=None):
    """造一个 torch.save 风格的 zip checkpoint；给了 epoch 就在 data.pkl 里写出真实的 pickle，否则内容不是合法 pickle。"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps({"epoch": epoch, "model": {"w": [0.0] * 10}}, protocol=2)
                   if epoch is not None else "x" * 1000)
    if not valid:  # 模拟写到一半被打断：砍掉文件尾的 zip 目录
        data = path.read_bytes()
        path.write_bytes(data[:len(data) // 2])


class ResumeHelperTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_find_checkpoint_prefers_latest_valid(self):
        self.assertIsNone(find_checkpoint(self.d))
        make_ckpt(self.d / "checkpoint0019.pth", epoch=19)
        make_ckpt(self.d / "checkpoint0039.pth", epoch=39)
        make_ckpt(self.d / "checkpoint_best.pth", epoch=99)  # 不是训练进度，不应被选中
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0039.pth")
        make_ckpt(self.d / "checkpoint.pth", epoch=44)
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint.pth")

    def test_find_checkpoint_skips_truncated(self):
        make_ckpt(self.d / "checkpoint0019.pth", epoch=19)
        make_ckpt(self.d / "checkpoint.pth", valid=False, epoch=25)
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0019.pth")

    def test_checkpoint_pth_older_than_numbered_is_not_preferred(self):
        # checkpoint.pth 不再无条件优先：它的 epoch 比编号文件旧时（如被旧运行遗留）应选编号文件
        make_ckpt(self.d / "checkpoint.pth", epoch=5)
        make_ckpt(self.d / "checkpoint0019.pth", epoch=19)
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0019.pth")

    def test_same_epoch_prefers_newer_mtime(self):
        make_ckpt(self.d / "checkpoint.pth", epoch=19)
        make_ckpt(self.d / "checkpoint0019.pth", epoch=19)
        os.utime(self.d / "checkpoint0019.pth", (1000, 1000))
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint.pth")
        os.utime(self.d / "checkpoint.pth", (500, 500))
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0019.pth")

    def test_numbered_files_sorted_numerically_not_as_strings(self):
        # 内容读不出 epoch 时按文件名里的数字比较："checkpoint100" 的字符串序比 "checkpoint99" 小
        make_ckpt(self.d / "checkpoint99.pth")
        make_ckpt(self.d / "checkpoint100.pth")
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint100.pth")

    def test_unreadable_epoch_falls_back_to_mtime(self):
        make_ckpt(self.d / "checkpoint.pth")  # 读不出 epoch
        make_ckpt(self.d / "checkpoint0019.pth", epoch=19)
        os.utime(self.d / "checkpoint0019.pth", (1000, 1000))
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint.pth")
        os.utime(self.d / "checkpoint.pth", (500, 500))
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0019.pth")

    def test_reading_epoch_never_executes_pickle_code(self):
        marker = self.d / "pwned"

        class Evil:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        with zipfile.ZipFile(self.d / "checkpoint.pth", "w") as z:
            z.writestr("archive/data.pkl", pickle.dumps({"epoch": 7, "x": Evil()}, protocol=2))
        make_ckpt(self.d / "checkpoint0003.pth", epoch=3)
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint.pth")  # epoch 7 照常读出
        self.assertFalse(marker.exists())

    def test_with_resume(self):
        p = Path("/a b/ck.pth")
        self.assertEqual(with_resume("bash a.sh --batch_size 4", p), "bash a.sh --batch_size 4 --resume '/a b/ck.pth'")
        self.assertEqual(with_resume("bash a.sh --resume old.pth --seed 1", p), "bash a.sh --resume '/a b/ck.pth' --seed 1")
        self.assertEqual(with_resume("bash a.sh --resume=old.pth", p), "bash a.sh --resume '/a b/ck.pth'")


class LockTest(unittest.TestCase):
    def test_one_monitor_per_window(self):
        state = Path(tempfile.mkdtemp())
        h = acquire_lock(state, "w1")
        with self.assertRaises(AlreadyRunning) as cm:
            acquire_lock(state, "w1")
        self.assertEqual(cm.exception.pid, os.getpid())
        other = acquire_lock(state, "w2")  # 不同窗口互不影响
        self.assertEqual(running_monitors(state), {"w1": os.getpid(), "w2": os.getpid()})
        h.close()
        other.close()
        self.assertEqual(running_monitors(state), {})  # 进程/句柄结束后锁自动释放


class TmuxTimeoutTest(unittest.TestCase):
    """tmux 命令卡住（曾有 send-keys 卡 17 小时）时，监控不能跟着无声冻结。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        fake = self.dir / "fake-tmux"
        fake.write_text("#!/bin/sh\nsleep 30\n")
        fake.chmod(0o755)
        self.tm = Tmux(binary=str(fake), timeout=1)
        (self.dir / "task.md").write_text(f"# tmux: w1\nworkdir: {self.dir}\n## task-list\necho hi\n")
        self.out = []
        self.sup = Supervisor(self.dir / "task.md", self.dir / "state", self.dir / "logs", self.tm, "w1",
                              echo=self.out.append)

    def test_hung_tmux_raises_tmux_error_quickly(self):
        t0 = time.time()
        with self.assertRaises(TmuxError):
            self.tm.pane_pid("w1")
        self.assertLess(time.time() - t0, 5)

    def test_check_stops_monitor_when_tmux_does_not_respond(self):
        with self.assertRaises(MonitorStop) as cm:
            self.sup.check()
        self.assertIn("not respond", cm.exception.reason)

    def test_loop_returns_1_when_tmux_does_not_respond(self):
        self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 1)
        self.assertIn("not respond", "\n".join(self.out))


class SocketCleanupMixin:
    def kill_tmux(self):
        subprocess.run(["tmux", "-L", self.sock, "kill-server"], capture_output=True)
        Path(f"/tmp/tmux-{os.getuid()}/{self.sock}").unlink(missing_ok=True)  # kill-server 不删 socket 文件


class TmuxCase(SocketCleanupMixin, unittest.TestCase):
    """用独立 socket 的真实 tmux server 跑端到端流程；每个测试独立 server，避免 kill-server 后立即复用 socket 的竞态。"""

    def setUp(self):
        self.sock = f"tmuxsup-test-{os.getpid()}-{time.time_ns()}"
        self.tm = Tmux(self.sock)
        self.dir = Path(tempfile.mkdtemp())
        self.out = []  # 监控程序打印到终端的内容
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-s", "w1", "-x", "200", "bash", "--norc"], check=True)
        self.addCleanup(self.kill_tmux)
        self.sup = self.make()

    def make(self, **kw):
        kw = {"retry_delay": 0.2, "start_grace": 1, "interrupt_wait": 0.3, **kw}  # 手动中断后的等待在这里缩短，免得每个用例等 30s
        return Supervisor(self.dir / "task.md", self.dir / "state", self.dir / "logs", self.tm, "w1",
                          echo=self.out.append, **kw)

    def write_tasks(self, body, extra=""):
        (self.dir / "task.md").write_text(f"# tmux: w1\nworkdir: {self.dir}\n{extra}\n{body}")

    def win(self):
        return load_windows(self.dir / "task.md")["w1"]

    def drive(self, cond, timeout=40):
        end = time.time() + timeout
        while time.time() < end:
            self.sup.check()
            if cond():
                return
            time.sleep(0.3)
        self.fail(f"超时；状态={self.sup.load_state()}；输出={self.out}")

    def drive_until_stop(self, timeout=40) -> MonitorStop:
        end = time.time() + timeout
        while time.time() < end:
            try:
                self.sup.check()
            except MonitorStop as e:
                return e
            time.sleep(0.3)
        self.fail(f"超时，监控没有结束；状态={self.sup.load_state()}")

    def text(self):
        return "\n".join(self.out)


class QueueTest(TmuxCase):
    def test_task_stays_at_head_until_success_then_moves_to_worked(self):
        self.write_tasks("## task-list\nsleep 3; echo A >> out\necho B >> out\n")
        self.sup.check()  # 启动 A
        self.assertEqual(self.win().tasks[0], "sleep 3; echo A >> out")  # 运行中仍在首行
        self.assertEqual(self.win().worked, [])
        self.drive(lambda: (self.dir / "out").exists() and (self.dir / "out").read_text() == "A\nB\n"
                   and self.sup.load_state()["run_id"] is None)
        self.assertEqual(self.win().tasks, [])
        self.assertEqual(plain(self.win().worked), ["sleep 3; echo A >> out", "echo B >> out"])

    def test_worked_entry_records_log_and_output_dir(self):
        script = self.dir / "o.sh"
        script.write_text("echo \"Namespace(output_dir='od')\"\n")
        self.write_tasks(f"## task-list\nbash {script}\nbash -c 'true'\n")
        self.drive(lambda: len(self.win().worked) == 2)
        with_out, plain = self.win().worked
        log = sorted((self.dir / "logs").glob("w1-*.log"))[0]
        self.assertEqual(with_out, f"bash {script}  # [done log={log} output_dir={self.dir}/od]")
        self.assertRegex(plain, r"^bash -c 'true'  # \[done log=\S+\.log\]$")  # 日志里没有 output_dir 时只记日志位置

    def test_per_task_workdir_does_not_leak_to_following_tasks(self):
        (self.dir / "sub").mkdir()
        self.write_tasks(f"## task-list\n[workdir=sub] pwd >> {self.dir}/out\npwd >> {self.dir}/out\n")  # 用绝对路径写结果，因为第一条的 cwd 是 sub
        self.drive(lambda: (self.dir / "out").exists() and len((self.dir / "out").read_text().split()) == 2
                   and self.sup.load_state()["run_id"] is None)
        self.assertEqual((self.dir / "out").read_text().split(),
                         [str((self.dir / "sub").resolve()), str(self.dir.resolve())])

    def test_empty_queue_finishes_monitor(self):
        self.write_tasks("## task-list\n")
        self.sup.check()
        self.assertTrue(self.sup.finished)
        self.assertIn("all tasks finished", self.text())

    def test_not_finished_while_a_task_is_still_running(self):
        self.write_tasks("## task-list\nsleep 3\n")
        self.sup.check()
        self.assertFalse(self.sup.finished)  # 最后一条在运行时队列虽然还有它自己，但绝不能提前判定完成
        self.drive(lambda: self.sup.finished)
        self.assertEqual(self.win().tasks, [])


class RetryTest(TmuxCase):
    def test_oom_halves_batch_other_errors_keep_batch(self):
        script = self.dir / "t.sh"
        # --batch_size>2 报 CUDA OOM；bs=2 时第一次报 IO 错误，再之后成功
        script.write_text(
            'BATCH_SIZE=$2\n'
            'echo "bs=$BATCH_SIZE" >> seen\n'
            'if [ "$BATCH_SIZE" -gt 2 ]; then echo "RuntimeError: CUDA out of memory." >&2; exit 1; fi\n'
            'if [ ! -f flag ]; then touch flag; echo "OSError: read failure" >&2; exit 1; fi\n'
            'echo ok >> seen\n')
        self.write_tasks(f"## task-list\nbash {script} --batch_size 8\n")
        self.drive(lambda: (self.dir / "seen").exists() and "ok" in (self.dir / "seen").read_text())
        self.assertEqual((self.dir / "seen").read_text().split(), ["bs=8", "bs=4", "bs=2", "bs=2", "ok"])

    def _argparse_like_script(self, name, ignore_args=False, default_bs=8):
        """模拟用户的训练脚本：batch_size 默认值写在脚本里，额外参数追加在最后（argparse 以最后一个为准），
        并像 main.py 一样把 Namespace(batch_size=N) 打到日志。ignore_args=True 模拟不转发 "$@" 的脚本。"""
        script = self.dir / name
        script.write_text(
            f'BATCH_SIZE="${{BATCH_SIZE:-{default_bs}}}"\n'
            f'args=(--batch_size "$BATCH_SIZE" {"" if ignore_args else '"$@"'})\n'
            'bs=0; for ((i=0; i<${#args[@]}; i++)); do [ "${args[$i]}" = --batch_size ] && bs=${args[$((i+1))]}; done\n'
            'echo "Namespace(batch_size=$bs, batch_size_total=8, seed=1)"\n'
            'echo "bs=$bs" >> seen\n'
            'if [ "$bs" -gt 2 ]; then echo "RuntimeError: CUDA out of memory." >&2; exit 1; fi\n'
            'echo ok >> seen\n')
        return script

    def test_oom_with_batch_size_only_in_script_appends_override(self):
        script = self._argparse_like_script("t2.sh")
        self.write_tasks(f"## task-list\nbash {script}\n")  # 任务行里没有 --batch_size，默认值 8 在脚本里
        self.drive(lambda: (self.dir / "seen").exists() and "ok" in (self.dir / "seen").read_text())
        self.assertEqual((self.dir / "seen").read_text().split(), ["bs=8", "bs=4", "bs=2", "ok"])
        self.assertIn("CUDA OOM: --batch_size 8 -> 4", self.text())
        self.assertEqual(self.win().failed, [])

    def test_oom_when_script_ignores_batch_override_fails_with_reason(self):
        script = self._argparse_like_script("t3.sh", ignore_args=True)
        self.write_tasks(f"## task-list\nbash {script}\n")
        self.drive(lambda: bool(self.win().failed))
        self.assertEqual((self.dir / "seen").read_text().split(), ["bs=8", "bs=8"])  # 只空转一次就判失败
        self.assertIn("does not apply the --batch_size override", self.win().failed[0])

    def test_oom_with_batch_size_1_in_log_goes_to_failed_list(self):
        script = self._argparse_like_script("t4.sh", default_bs=1)
        script.write_text(script.read_text().replace('-gt 2', '-ge 1'))  # bs=1 也 OOM
        self.write_tasks(f"## task-list\nbash {script}\n")
        self.drive(lambda: bool(self.win().failed))
        self.assertEqual((self.dir / "seen").read_text().split(), ["bs=1"])
        self.assertIn("no reducible --batch_size", self.win().failed[0])

    def _accum_script(self, name, default_bs=8, default_acc=1):
        """batch_size / accum_steps 默认值写在脚本里、额外参数追加在最后的训练脚本；bs>2 时报 CUDA OOM。
        每次运行把 "bs=N acc=M" 记到 seen，并像 main.py 一样把两者打到日志的 Namespace 里。"""
        script = self.dir / name
        script.write_text(
            f'args=(--batch_size {default_bs} --accum_steps {default_acc} "$@")\n'
            'bs=0; acc=0\n'
            'for ((i=0; i<${#args[@]}; i++)); do\n'
            '  [ "${args[$i]}" = --batch_size ] && bs=${args[$((i+1))]}\n'
            '  [ "${args[$i]}" = --accum_steps ] && acc=${args[$((i+1))]}\n'
            'done\n'
            'echo "Namespace(batch_size=$bs, accum_steps=$acc, seed=1)"\n'
            'echo "bs=$bs acc=$acc" >> seen\n'
            'if [ "$bs" -gt 2 ]; then echo "torch.cuda.OutOfMemoryError: CUDA out of memory." >&2; exit 1; fi\n'
            'echo ok >> seen\n')
        return script

    def _seen(self):
        return (self.dir / "seen").read_text().split("\n")[:-1]

    def test_oom_halving_batch_doubles_accum_steps_when_both_in_script(self):
        script = self._accum_script("a1.sh")  # 任务行里两个参数都没有，值在脚本里
        self.write_tasks(f"## task-list\nbash {script}\n")
        self.drive(lambda: (self.dir / "seen").exists() and "ok" in self._seen())
        self.assertEqual(self._seen(), ["bs=8 acc=1", "bs=4 acc=2", "bs=2 acc=4", "ok"])  # 逻辑 batch 恒为 8
        self.assertIn("--accum_steps 1 -> 2", self.text())

    def test_oom_rewrites_accum_steps_in_command_in_place(self):
        script = self._accum_script("a2.sh")
        self.write_tasks(f"## task-list\nbash {script} --batch_size 8 --accum_steps 3\n")
        self.drive(lambda: (self.dir / "seen").exists() and "ok" in self._seen())
        self.assertEqual(self._seen(), ["bs=8 acc=3", "bs=4 acc=6", "bs=2 acc=12", "ok"])
        starts = [l for l in self.out if "start:" in l]
        self.assertTrue(all(l.count("--accum_steps") == 1 and l.count("--batch_size") == 1 for l in starts), starts)

    def test_oom_batch_in_command_accum_only_in_script_appends_accum(self):
        script = self._accum_script("a3.sh", default_acc=2)
        self.write_tasks(f"## task-list\nbash {script} --batch_size 8\n")
        self.drive(lambda: (self.dir / "seen").exists() and "ok" in self._seen())
        self.assertEqual(self._seen(), ["bs=8 acc=2", "bs=4 acc=4", "bs=2 acc=8", "ok"])

    def test_oom_odd_batch_rounds_accum_up(self):
        script = self._accum_script("a4.sh", default_bs=5)
        self.write_tasks(f"## task-list\nbash {script}\n")
        self.drive(lambda: (self.dir / "seen").exists() and "ok" in self._seen())
        self.assertEqual(self._seen(), ["bs=5 acc=1", "bs=2 acc=3", "ok"])  # 5//2=2，accum=ceil(5/2)=3（逻辑 batch 6，略增大）

    def test_oom_regex_covers_common_cuda_oom_formats(self):
        for msg in [
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB",
            "torch.cuda.OutOfMemoryError: ",
            "OutOfMemoryError",
            "RuntimeError: CUDA error: out of memory",
            "RuntimeError: CUDA driver error: out of memory",
            "CUDA_ERROR_OUT_OF_MEMORY: out of memory",
            "cudaErrorMemoryAllocation: out of memory",
            "RuntimeError: CUDA error: cudaErrorMemoryAllocation",
            "Tried to allocate 20.00 MiB (GPU 0; 15.78 GiB total capacity; 14.2 GiB already allocated)",
            "RuntimeError: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
            "RuntimeError: cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
            "RuntimeError: cusolver error: CUSOLVER_STATUS_ALLOC_FAILED",
            "torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 1.00 GiB",
            "hipErrorOutOfMemory",
            "NCCL error: unhandled cuda error, NCCL version 2.18, out of memory",
        ]:
            with self.subTest(msg=msg):
                self.assertTrue(OOM_RE.search(msg), msg)

    def test_oom_regex_ignores_non_oom_errors(self):
        for msg in [
            "RuntimeError: CUDA error: an illegal memory access was encountered",
            "RuntimeError: CUDA error: device-side assert triggered",
            "RuntimeError: DefaultCPUAllocator: can't allocate memory: you tried to allocate 1000 bytes",
            "MemoryError",
            "Killed",
            "Memory usage: 12 GiB, out of 80 GiB",
            "CUDA out of sync",
            "KeyError: 'memory'",
        ]:
            with self.subTest(msg=msg):
                self.assertFalse(OOM_RE.search(msg), msg)

    def test_exhausted_retries_go_to_failed_list_and_queue_continues(self):
        self.sup = self.make(max_retries=1)
        self.write_tasks("## task-list\nbash -c 'echo x >> n; exit 3'\necho next >> n\n")
        self.drive(lambda: (self.dir / "n").exists() and "next" in (self.dir / "n").read_text()
                   and self.sup.load_state()["run_id"] is None)
        self.assertEqual((self.dir / "n").read_text().split(), ["x", "x", "next"])  # 首次 + 1 次重试，然后继续下一条
        w = self.win()
        self.assertEqual(w.failed, ["bash -c 'echo x >> n; exit 3'  # [failed exceeded max_retries=1 (exit code 3)]"])
        self.assertEqual(plain(w.worked), ["echo next >> n"])
        self.assertIn("exceeded max_retries=1", self.text())  # 原因打印到终端

    def test_oom_without_batch_arg_goes_to_failed_list(self):
        self.write_tasks("## task-list\nbash -c 'echo CUDA out of memory >&2; exit 1'\n")
        self.drive(lambda: bool(self.win().failed))
        self.assertIn("--batch_size", self.win().failed[0])
        self.assertEqual(self.win().tasks, [])

    def test_shell_syntax_error_goes_to_failed_list_and_queue_continues(self):
        self.write_tasks("## task-list\necho 'unclosed\necho ok >> out\n")
        self.drive(lambda: (self.dir / "out").exists())
        self.assertEqual(self.win().failed, ["echo 'unclosed  # [failed shell syntax error]"])
        self.assertIn("shell syntax error", self.text())

    def test_retry_waits_retry_delay(self):
        self.sup = self.make(max_retries=1, retry_delay=1.5)
        stamps = []
        orig = self.sup.echo
        self.sup.echo = lambda s: (stamps.append(time.time()) if "start:" in s else None, orig(s))[1]
        self.write_tasks("## task-list\nbash -c 'exit 3'\n")
        self.drive(lambda: bool(self.win().failed))
        self.assertEqual(len(stamps), 2)
        self.assertGreaterEqual(stamps[1] - stamps[0], 1.4)


class ResumeTest(TmuxCase):
    def _script(self, with_ckpt: bool):
        # 第一次运行：打印 Namespace(output_dir=...)，可选地写出 checkpoint，然后失败；第二次运行：记录收到的参数
        script = self.dir / "r.sh"
        mk = ('python3 -c "import zipfile;zipfile.ZipFile(\'od/checkpoint.pth\',\'w\').writestr(\'a\',\'b\')"\n'
              if with_ckpt else '')
        script.write_text(
            'if [ ! -f flag ]; then touch flag; mkdir -p od\n'
            "echo \"Namespace(output_dir='od')\"\n" + mk + 'echo "OSError: boom" >&2; exit 1; fi\n'
            'echo "args: $@" >> seen\n')
        self.write_tasks(f"## task-list\nbash {script}\n")

    def test_retry_resumes_from_checkpoint(self):
        self._script(with_ckpt=True)
        self.drive(lambda: (self.dir / "seen").exists())
        self.assertEqual((self.dir / "seen").read_text().strip(), f"args: --resume {self.dir}/od/checkpoint.pth")

    def test_retry_without_checkpoint_reruns_as_is(self):
        self._script(with_ckpt=False)
        self.drive(lambda: (self.dir / "seen").exists())
        self.assertEqual((self.dir / "seen").read_text().strip(), "args:")

    def test_no_auto_resume_flag(self):
        self.sup = self.make(auto_resume=False)
        self._script(with_ckpt=True)
        self.drive(lambda: (self.dir / "seen").exists())
        self.assertEqual((self.dir / "seen").read_text().strip(), "args:")


class ManualInterruptTest(TmuxCase):
    """手动退出：任务移入 failed list 并继续下一条；只有窗口丢失这类无法继续的情况才结束监控（见 AutoStopTest）。"""

    def assert_manual_then_next(self, exit_code):
        """任务被判为手动中断（而不是重试耗尽后失败）：failed 原因含 manually interrupted，且没有重试过。"""
        self.assert_failed_then_next(f"manually interrupted (exit code {exit_code})")
        self.assertNotIn("retry", self.text())

    def assert_failed_then_next(self, reason_part):
        self.drive(lambda: len(self.win().worked) == 1)  # 下一条已启动并成功
        self.assertTrue((self.dir / "next").exists())
        self.assertEqual(self.win().tasks, [])
        self.assertEqual(len(self.win().worked), 1)
        self.assertEqual(len(self.win().failed), 1)
        self.assertIn(reason_part, self.win().failed[0])

    def log_text(self):
        return "".join(p.read_text() for p in (self.dir / "logs").glob("*.log"))

    def test_ctrl_c_moves_task_to_failed_and_runs_next(self):
        self.write_tasks("## task-list\nsleep 60\necho x > next\n")
        self.sup.check()
        time.sleep(1)
        subprocess.run(["tmux", "-L", self.sock, "send-keys", "-t", "=w1:", "C-c"], check=True)
        self.assert_failed_then_next("manually interrupted")

    def test_exit_code_130(self):
        self.write_tasks("## task-list\nbash -c 'exit 130'\necho x > next\n")
        self.assert_manual_then_next(130)

    def test_exit_code_143(self):
        self.write_tasks("## task-list\nbash -c 'exit 143'\necho x > next\n")
        self.assert_manual_then_next(143)

    def test_ctrl_c_on_task_that_handles_sigint_is_manual_and_keeps_log(self):
        # 模拟 torchrun：捕获 SIGINT 后继续打印收尾信息，再以 1 退出。
        # tee 若被 Ctrl-C 一并杀死，Python 退出时 flush 失败得到 120，且收尾输出（KeyboardInterrupt）丢失，
        # 任务会被误当普通错误重试；wrapper 里的 tee 必须忽略 SIGINT。
        script = self.dir / "graceful.py"
        script.write_text(
            "import sys, time\n"
            "print('started', flush=True)\n"
            "try:\n"
            "    time.sleep(60)\n"
            "except KeyboardInterrupt:\n"
            "    print('KeyboardInterrupt: cleanup')\n"
            "    sys.exit(1)\n")
        self.write_tasks(f"## task-list\npython3 {script}\necho x > next\n")
        self.sup.check()
        self.drive(lambda: "started" in self.log_text())  # 等脚本装好 SIGINT 处理再发 Ctrl-C
        subprocess.run(["tmux", "-L", self.sock, "send-keys", "-t", "=w1:", "C-c"], check=True)
        self.assert_manual_then_next(1)
        self.assertIn("KeyboardInterrupt: cleanup", self.log_text())

    def test_nonzero_exit_with_keyboard_interrupt_in_log_counts_as_manual(self):
        self.write_tasks("## task-list\nbash -c 'echo KeyboardInterrupt >&2; exit 1'\necho x > next\n")
        self.assert_manual_then_next(1)

    def test_torchrun_got_signal_2_in_log_counts_as_manual(self):
        self.write_tasks("## task-list\nbash -c 'echo SignalException: Process 7 got signal: 2 >&2; exit 1'\necho x > next\n")
        self.assert_manual_then_next(1)

    def test_torchrun_got_signal_15_in_log_counts_as_manual(self):
        self.write_tasks("## task-list\nbash -c 'echo SignalException: Process 7 got signal: 15 >&2; exit 1'\necho x > next\n")
        self.assert_manual_then_next(1)

    def test_120_with_keyboard_interrupt_in_log_counts_as_manual(self):
        self.write_tasks("## task-list\nbash -c 'echo KeyboardInterrupt >&2; exit 120'\necho x > next\n")
        self.assert_manual_then_next(120)

    def test_120_without_keyboard_interrupt_is_retried(self):
        self.sup = self.make(max_retries=1)
        self.write_tasks("## task-list\nbash -c 'exit 120'\necho x > next\n")
        self.assert_failed_then_next("exceeded max_retries=1 (exit code 120)")
        self.assertIn("retrying unchanged", self.text())

    def test_idle_window_without_exit_record_moves_task_to_failed_and_runs_next(self):
        self.write_tasks("## task-list\nsleep 60\necho x > next\n")
        self.sup.check()
        time.sleep(1)
        subprocess.run(["pkill", "-KILL", "-P", str(self.tm.pane_pid("w1"))])  # 连子 shell 带 tee 一起杀掉，退出记录不会被写出
        self.assert_failed_then_next("no exit record")


class InterruptWaitTest(TmuxCase):
    """判为手动中断时先等一段时间（默认 30s）再移入 failed list 并启动下一条；等待期间 stop 可保住队列。"""

    WAIT = 4

    def start_checks(self):
        """在线程里不停 check（等待是阻塞的），并记下"进入等待"和"移入 failed"两个事件的时间。"""
        self.stamps = {}
        orig = self.sup.echo

        def echo(s):
            for key, mark in (("wait", "waiting"), ("failed", "FAILED")):
                if mark in s:
                    self.stamps.setdefault(key, time.time())
            orig(s)
        self.sup.echo = echo

        def loop():
            end = time.time() + 40
            while time.time() < end and not self.sup._stop and "failed" not in self.stamps:
                try:
                    self.sup.check()
                except MonitorStop:
                    return
                time.sleep(0.3)
        self.thread = threading.Thread(target=loop)
        self.thread.start()
        self.addCleanup(self.thread.join, 10)

    def wait_for(self, cond, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.1)
        self.fail(f"超时；输出={self.out}")

    def assert_waits_before_failing(self):
        self.wait_for(lambda: "wait" in self.stamps)
        # 等待期间：任务还在 task-list 首行，failed list 为空，下一条没有启动
        self.assertEqual(self.win().failed, [])
        self.assertTrue(self.win().tasks[0].startswith(self.first))
        self.assertFalse((self.dir / "next").exists())
        self.wait_for(lambda: "failed" in self.stamps)
        self.assertGreaterEqual(self.stamps["failed"] - self.stamps["wait"], self.WAIT - 0.2)
        self.wait_for(lambda: (self.dir / "next").exists())
        self.assertEqual(len(self.win().failed), 1)

    def test_exit_code_130_waits_before_moving_to_failed(self):
        self.sup = self.make(interrupt_wait=self.WAIT)
        self.first = "bash -c 'exit 130'"
        self.write_tasks(f"## task-list\n{self.first}\necho x > next\n")
        self.start_checks()
        self.assert_waits_before_failing()

    def test_keyboard_interrupt_in_log_waits_before_moving_to_failed(self):
        self.sup = self.make(interrupt_wait=self.WAIT)
        self.first = "bash -c 'echo KeyboardInterrupt >&2; exit 1'"
        self.write_tasks(f"## task-list\n{self.first}\necho x > next\n")
        self.start_checks()
        self.assert_waits_before_failing()

    def test_idle_window_without_exit_record_waits_before_moving_to_failed(self):
        self.sup = self.make(interrupt_wait=self.WAIT)
        self.first = "sleep 60"
        self.write_tasks(f"## task-list\n{self.first}\necho x > next\n")
        self.sup.check()
        time.sleep(1.2)  # 过了启动宽限期
        subprocess.run(["pkill", "-KILL", "-P", str(self.tm.pane_pid("w1"))])
        self.start_checks()
        self.assert_waits_before_failing()

    def test_stop_during_wait_keeps_task_in_queue(self):
        self.sup = self.make(interrupt_wait=30)
        self.write_tasks("## task-list\nbash -c 'exit 130'\necho x > next\n")
        self.start_checks()
        self.wait_for(lambda: "wait" in self.stamps)
        self.sup.request_stop()
        self.thread.join(10)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.win().failed, [])  # 没被移入 failed
        self.assertEqual(self.win().tasks, ["bash -c 'exit 130'", "echo x > next"])  # 队列原样保留
        self.assertFalse((self.dir / "next").exists())

    def test_ordinary_failure_does_not_wait(self):
        self.sup = self.make(interrupt_wait=60, max_retries=0)
        self.write_tasks("## task-list\nbash -c 'exit 3'\necho x > next\n")
        self.drive(lambda: (self.dir / "next").exists(), timeout=15)
        self.assertEqual(len(self.win().failed), 1)


class ExitRecordRaceTest(TmuxCase):
    def test_task_finishing_between_exit_check_and_idle_check_is_not_misjudged_as_interrupted(self):
        # 监控刚看到"没有退出记录"、还没判断忙闲时，任务恰好写出退出记录并结束：窗口空闲但退出记录已在，不能判成手动中断
        self.write_tasks("## task-list\nsleep 60\necho x > next\n")
        self.sup.check()
        time.sleep(1.2)  # 过了启动宽限期
        exit_file = Path(self.sup.load_state()["exit_file"])
        real_busy = self.tm.busy

        def busy_after_task_finished(pid, name=None):
            exit_file.write_text("0\n")  # 任务在这一瞬间写完退出记录并退出
            return False
        self.tm.busy = busy_after_task_finished
        self.addCleanup(setattr, self.tm, "busy", real_busy)
        self.sup.check()
        self.assertNotIn("manually interrupted", self.text())
        self.assertEqual(self.win().failed, [])
        self.assertFalse((self.dir / "next").exists())


class AutoStopTest(TmuxCase):
    """窗口丢失等无法继续的情况：监控自动结束（MonitorStop），任务留在 task-list 首行。"""

    def test_missing_tmux_window_stops_monitor(self):
        self.write_tasks("## task-list\necho hi\n")
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-s", "keep", "bash", "--norc"], check=True)
        subprocess.run(["tmux", "-L", self.sock, "kill-session", "-t", "=w1"], check=True)
        with self.assertRaises(MonitorStop) as cm:
            self.sup.check()
        self.assertIn("does not exist", cm.exception.reason)


class ExternalCommandTest(TmuxCase):
    def test_waits_for_manually_started_command_and_prints_notice(self):
        self.write_tasks("## task-list\necho queued >> out\n")
        self.tm.send("w1", "sleep 4")  # 用户手动在窗口里跑的命令
        time.sleep(0.5)
        self.sup.check()
        self.assertEqual(self.sup.load_state()["status"], "external")
        self.assertIn("not started by this tool", self.text())
        self.assertFalse((self.dir / "out").exists())
        self.drive(lambda: (self.dir / "out").exists())

    def test_no_keys_sent_to_external_command_in_plain_window(self):
        self.write_tasks("## task-list\necho queued >> out\n")
        self.tm.send("w1", "sleep 3")
        time.sleep(0.5)
        self.sup.check()
        text = self.tm._run("capture-pane", "-p", "-t", "=w1:").stdout
        self.assertNotIn("^A", text)
        self.assertNotIn("^C", text)


class NestedShellTest(TmuxCase):
    """窗口里套了 srun/salloc/ssh 这类转接进程（如 `srun --pty bash` 进计算节点）：看嵌套 shell 的提示符判断忙闲。"""

    def setUp(self):
        super().setUp()
        srun = self.dir / "srun"  # 进程名为 srun 的 bash，模拟转接进程
        srun.write_bytes(Path("/bin/bash").read_bytes())
        srun.chmod(0o755)
        self.tm.send("w1", f"PS1='nested$ ' {srun} --norc")
        time.sleep(1)

    def busy(self):
        return self.tm.busy(self.tm.pane_pid("w1"), "w1")

    def type_raw(self, text):
        self.tm._run("send-keys", "-t", self.tm.target("w1"), "-l", "--", text)
        time.sleep(0.5)

    def test_idle_at_prompt_and_busy_while_command_runs(self):
        self.assertFalse(self.busy())
        self.tm.send("w1", "sleep 3")
        time.sleep(0.8)
        self.assertTrue(self.busy())
        time.sleep(3)
        self.assertFalse(self.busy())

    def test_queue_runs_in_nested_shell(self):
        self.write_tasks("## task-list\necho one\necho two")
        self.drive(lambda: self.win().worked and len(self.win().worked) == 2, timeout=60)

    def test_leftover_ending_with_prompt_char_in_nested_shell_is_cleared(self):
        self.type_raw("garbage $")  # 光标行仍以提示符字符结尾，被判为空闲；清行用 C-a C-k 而不是 C-c
        self.write_tasks("## task-list\necho ok >> out\n")
        self.drive(lambda: (self.dir / "out").exists() and self.sup.load_state()["run_id"] is None)
        self.assertEqual((self.dir / "out").read_text(), "ok\n")
        self.assertEqual(self.win().failed, [])

    def pane_text(self):
        return self.tm._run("capture-pane", "-p", "-t", "=w1:").stdout

    def test_leftover_in_nested_shell_is_cleared_before_launch(self):
        # 残留文字让光标行不以提示符结尾，看起来像在忙；要发新命令时先 C-a C-k 清一次再重判，清掉后照常启动
        self.type_raw("garbage")
        self.write_tasks("## task-list\necho ok >> out\n")
        self.drive(lambda: (self.dir / "out").exists() and self.sup.load_state()["run_id"] is None)
        self.assertEqual((self.dir / "out").read_text(), "ok\n")
        self.assertEqual(self.win().failed, [])

    def test_running_nested_program_is_kept_waiting_and_keys_sent_only_once(self):
        self.tm.send("w1", f"sleep 6; echo alive > {self.dir}/alive")
        time.sleep(0.8)
        self.write_tasks("## task-list\necho ok >> out\n")
        for _ in range(3):  # 多轮检查：只有刚进入等待那一次会发 C-a C-k
            self.sup.check()
            time.sleep(0.3)
        self.assertEqual(self.sup.load_state()["status"], "external")
        self.assertEqual(self.pane_text().count("^A^K"), 1)
        self.assertFalse((self.dir / "out").exists())
        self.drive(lambda: (self.dir / "out").exists(), timeout=40)  # 程序结束后任务才启动
        self.assertTrue((self.dir / "alive").exists())  # 程序没被打断

    def test_nothing_is_sent_to_nested_program_when_queue_is_empty(self):
        self.tm.send("w1", "sleep 3")
        time.sleep(0.8)
        self.write_tasks("## task-list\n")
        self.sup.check()
        self.assertNotIn("^A", self.pane_text())

    def test_clear_input_sends_nothing_while_nested_command_runs(self):
        self.tm.send("w1", f"sleep 3; echo alive > {self.dir}/alive")
        time.sleep(0.8)
        self.assertFalse(self.tm.clear_input("w1", self.tm.pane_pid("w1")))
        time.sleep(3.5)
        self.assertTrue((self.dir / "alive").exists())

    def test_program_misjudged_idle_is_not_killed_by_clear_input(self):
        # 输出行以 % 结尾（如进度 "45%"）会被当成提示符而误判空闲；这时发出的清行按键不能杀掉程序
        self.tm.send("w1", f"printf 'progress 45%%'; sleep 3; echo alive > {self.dir}/alive")
        time.sleep(0.8)
        self.assertFalse(self.busy())  # 确认确实被误判为空闲
        self.assertTrue(self.tm.clear_input("w1", self.tm.pane_pid("w1")))
        time.sleep(3.5)
        self.assertTrue((self.dir / "alive").exists())


class CopyModeTest(TmuxCase):
    """用户在窗口里滚动查看历史（copy-mode）时，发送的命令不能被模式吞掉。"""

    def enter_copy_mode(self):
        subprocess.run(["tmux", "-L", self.sock, "copy-mode", "-t", "=w1:"], check=True)

    def in_mode(self):
        return self.tm._run("display-message", "-p", "-t", "=w1:", "#{pane_in_mode}").stdout.strip() == "1"

    def test_send_exits_copy_mode_first(self):
        self.enter_copy_mode()
        self.tm.send("w1", f"echo ok > {self.dir}/sent")
        end = time.time() + 5
        while time.time() < end and not (self.dir / "sent").exists():
            time.sleep(0.2)
        self.assertTrue((self.dir / "sent").exists())

    def test_clear_input_exits_copy_mode(self):
        self.enter_copy_mode()
        self.assertTrue(self.tm.clear_input("w1", self.tm.pane_pid("w1")))
        self.assertFalse(self.in_mode())

    def test_clear_input_leaves_copy_mode_alone_while_command_runs(self):
        self.tm.send("w1", "sleep 3")
        time.sleep(0.5)
        self.enter_copy_mode()
        self.assertFalse(self.tm.clear_input("w1", self.tm.pane_pid("w1")))  # 忙碌：什么都不动
        self.assertTrue(self.in_mode())


class DirtyInputTest(TmuxCase):
    """窗口 shell 里有残留字符时，发出的命令不能被它们污染；窗口里有外部程序在跑时一个按键都不能发。"""

    def type_raw(self, text, enter=False):
        t = self.tm.target("w1")
        self.tm._run("send-keys", "-t", t, "-l", "--", text)
        if enter:
            self.tm._run("send-keys", "-t", t, "C-m")
        time.sleep(0.5)

    def assert_queue_runs_cleanly(self):
        self.write_tasks("## task-list\necho ok >> out\n")
        self.drive(lambda: (self.dir / "out").exists() and self.sup.load_state()["run_id"] is None)
        self.assertEqual((self.dir / "out").read_text(), "ok\n")
        self.assertEqual(self.win().failed, [])
        self.assertEqual(plain(self.win().worked), ["echo ok >> out"])

    def test_leftover_text_on_prompt_is_cleared(self):
        self.type_raw("garbage")
        self.assert_queue_runs_cleanly()

    def test_unfinished_quote_continuation_is_cleared(self):
        self.type_raw("echo 'abc", enter=True)  # shell 停在续行提示符 `> `
        self.assert_queue_runs_cleanly()

    def run_external_sigint_probe(self):
        """在窗口里手动起一个外部程序：约 3 秒，收到 SIGINT 就记 got，正常结束记 end（用绝对路径，不依赖 cwd）。"""
        sig = self.dir / "sig"
        self.type_raw(f"bash -c 'trap \"echo got >> {sig}\" INT; for i in 1 2 3 4 5 6; do sleep 0.5; done; echo end >> {sig}'",
                      enter=True)
        return sig

    def test_clear_input_sends_nothing_while_a_command_runs(self):
        sig = self.run_external_sigint_probe()
        pid = self.tm.pane_pid("w1")
        self.assertTrue(self.tm.busy(pid))
        self.assertFalse(self.tm.clear_input("w1", pid))  # 忙碌：不发 C-c
        self.assertTrue(self.tm.wait_idle(pid, 10))
        self.assertEqual(sig.read_text(), "end\n")  # 没收到 SIGINT，跑到了结尾

    def test_external_command_survives_and_queue_waits(self):
        self.write_tasks("## task-list\necho queued >> out\n")
        sig = self.run_external_sigint_probe()
        self.sup.check()
        self.assertEqual(self.sup.load_state()["status"], "external")
        self.assertFalse((self.dir / "out").exists())
        self.drive(lambda: (self.dir / "out").exists())
        self.assertEqual(sig.read_text(), "end\n")  # 外部程序没被打断，队列等它结束后才启动

    def test_launch_while_a_command_started_after_idle_check_leaves_it_alone(self):
        # 竞态：监控以为窗口空闲，真正发键时窗口已被外部程序占用 -> 不发任何键，任务留在 task-list，之后当外部命令等待
        self.write_tasks("## task-list\necho queued >> out\n")
        sig = self.run_external_sigint_probe()
        self.sup._start_next(self.win(), self.sup.load_state())  # 跳过空闲检查直接启动
        self.assertIsNone(self.sup.load_state()["run_id"])
        self.assertEqual(self.win().tasks, ["echo queued >> out"])
        self.assertIn("not started by this tool", self.text())
        self.drive(lambda: (self.dir / "out").exists())
        self.assertEqual(sig.read_text(), "end\n")


class StartFreshTest(TmuxCase):
    def test_reset_discards_previous_state(self):
        self.write_tasks("## task-list\nsleep 30\n")
        self.sup.check()
        self.assertTrue(self.sup.load_state()["run_id"])
        self.sup.reset()  # run 启动时调用：新的监控不继承旧状态
        self.assertIsNone(self.sup.load_state()["run_id"])
        self.sup.check()  # 旧任务仍在窗口里运行 -> 当作外部命令等待，不会重复启动
        self.assertEqual(self.sup.load_state()["status"], "external")


class CtrlCResumeTest(TmuxCase):
    """Ctrl-C 监控进程只是暂离：状态保留，重新 run 接着监控；stop 则下次是全新的监控。"""

    def start_then_ctrl_c(self):
        self.write_tasks("## task-list\nsleep 3; echo A >> out\necho B >> out\n")
        with mock.patch.object(self.sup, "_sleep", side_effect=KeyboardInterrupt):
            self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 0)
        self.assertTrue(self.sup.load_state()["run_id"])

    def test_rerun_after_ctrl_c_resumes_the_running_task(self):
        self.start_then_ctrl_c()
        self.sup = self.make()  # 新的监控进程，沿用同一个状态目录
        self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 0)
        self.assertEqual((self.dir / "out").read_text().split(), ["A", "B"])  # A 只跑了一次
        self.assertEqual(len(self.win().worked), 2)
        self.assertNotIn("not started by this tool", self.text())

    def test_stop_clears_state(self):
        self.write_tasks("## task-list\nsleep 30\n")
        self.sup.check()
        self.sup.request_stop()
        self.sup.run_forever(interval=600, poll=0.2)
        self.assertIsNone(self.sup.load_state()["run_id"])


class RunLoopTest(TmuxCase):
    def test_request_stop_ends_loop_with_code_0(self):
        self.write_tasks("## task-list\nsleep 30\n")
        result = []
        t = threading.Thread(target=lambda: result.append(self.sup.run_forever(interval=600, poll=0.2)))
        t.start()
        time.sleep(2)
        self.sup.request_stop()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [0])
        self.assertIn("stop", self.text())

    def test_loop_notices_idle_window_without_waiting_for_interval(self):
        # 没有退出记录时（如 Ctrl-C 把子 shell 一并中断），不应等到下一次全量检查（interval=600s）才发现
        self.write_tasks("## task-list\nsleep 60\n")
        self.addCleanup(self.sup.request_stop)
        result = []
        t = threading.Thread(target=lambda: result.append(self.sup.run_forever(interval=600, poll=0.2)))
        t.start()
        time.sleep(2)
        subprocess.run(["pkill", "-KILL", "-P", str(self.tm.pane_pid("w1"))])
        t.join(15)
        self.assertFalse(t.is_alive(), "监控没有及时发现窗口已空闲")
        self.assertEqual(result, [0])  # 任务移入 failed list，队列跑空，监控正常退出
        self.assertIn("no exit record", self.text())

    def test_loop_exits_0_when_all_tasks_are_done(self):
        # 成功 -> 自动跑下一条 -> 队列跑空 -> 监控自己退出（退出码 0）；interval 很长，说明不是靠周期检查
        self.write_tasks("## task-list\necho A >> out\necho B >> out\n")
        self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 0)
        self.assertEqual((self.dir / "out").read_text().split(), ["A", "B"])
        self.assertEqual(len(self.win().worked), 2)
        self.assertIn("all tasks finished", self.text())

    def test_loop_exits_0_when_queue_ends_with_failures_and_reports_them(self):
        self.sup = self.make(max_retries=0)
        self.write_tasks("## task-list\nbash -c 'exit 3'\necho ok >> out\n")
        self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 0)
        self.assertIn("failed list: 1", self.text())

    def test_loop_exits_immediately_when_queue_is_empty_at_start(self):
        self.write_tasks("## task-list\n")
        self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 0)

    def test_loop_returns_1_on_auto_stop(self):
        self.write_tasks("## task-list\necho hi\n")
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-s", "keep", "bash", "--norc"], check=True)
        subprocess.run(["tmux", "-L", self.sock, "kill-session", "-t", "=w1"], check=True)
        self.assertEqual(self.sup.run_forever(interval=600, poll=0.2), 1)
        self.assertIn("does not exist", self.text())


class CliTest(SocketCleanupMixin, unittest.TestCase):
    def setUp(self):
        self.sock = f"tmuxsup-cli-{os.getpid()}-{time.time_ns()}"
        self.dir = Path(tempfile.mkdtemp())
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-s", "w1", "bash", "--norc"], check=True)
        self.addCleanup(self.kill_tmux)
        (self.dir / "task.md").write_text(f"# tmux: w1\nworkdir: {self.dir}\n## task-list\nsleep 60\n")
        self.common = ["--task-file", str(self.dir / "task.md"), "--state-dir", str(self.dir / "state"),
                       "--log-dir", str(self.dir / "logs"), "--socket", self.sock]

    def cli(self, *args, **kw):
        return subprocess.run([str(ROOT / "tmuxsup"), *args], capture_output=True, text=True, timeout=30, **kw)

    def start(self):
        p = subprocess.Popen([str(ROOT / "tmuxsup"), "run", "w1", "--interval", "600", "--poll", "0.3", *self.common],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(p.stdout.close)
        self.addCleanup(lambda: p.poll() is None and p.kill())  # 后注册先执行：先杀进程再关管道
        end = time.time() + 15
        while time.time() < end and "w1" not in running_monitors(self.dir / "state"):
            time.sleep(0.2)
        self.assertIn("w1", running_monitors(self.dir / "state"))
        return p

    def test_second_run_refused_then_stop_then_run_again(self):
        p = self.start()
        r = self.cli("run", "w1", *self.common)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already running", r.stdout + r.stderr)
        r = self.cli("stop", "w1", "--state-dir", str(self.dir / "state"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(p.wait(15), 0)
        self.assertIn("stop", p.stdout.read())
        self.assertEqual(running_monitors(self.dir / "state"), {})
        p2 = self.start()  # stop 之后可以重新 run
        self.cli("stop", "--all", "--state-dir", str(self.dir / "state"))
        self.assertEqual(p2.wait(15), 0)

    def test_status_marks_state_as_stale_when_monitor_is_not_running(self):
        r = self.cli("status", "w1", *self.common)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("monitor: not running", r.stdout)
        self.assertIn("task-list: 1", r.stdout)
        self.cli("once", "w1", *self.common)  # 启动 sleep 60 后监控就退出了，状态文件里仍记着 running
        r = self.cli("status", "w1", *self.common)
        self.assertIn("last recorded", r.stdout)  # 提示这是监控退出前最后记录的状态，不是实时的

    def test_run_exits_0_by_itself_after_the_queue_drains(self):
        (self.dir / "task.md").write_text(
            f"# tmux: w1\nworkdir: {self.dir}\n## task-list\necho a >> {self.dir}/out\necho b >> {self.dir}/out\n")
        r = self.cli("run", "w1", "--interval", "600", "--poll", "0.3", *self.common)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.dir / "out").read_text().split(), ["a", "b"])  # 第一条成功后自动跑了第二条
        self.assertIn("all tasks finished", r.stdout)
        self.assertEqual(running_monitors(self.dir / "state"), {})

    def test_stop_when_not_running(self):
        r = self.cli("stop", "w1", "--state-dir", str(self.dir / "state"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("No monitor is running", r.stdout + r.stderr)

    def leave_stale_state(self):
        """复现"监控早已退出、状态文件还停在运行中"：once 启动 sleep 60 后监控就退出了，再把任务杀掉让窗口空闲。"""
        self.cli("once", "w1", *self.common)
        self.assertTrue((self.dir / "state" / "w1.json").exists())
        pane = subprocess.run(["tmux", "-L", self.sock, "display-message", "-p", "-t", "=w1:", "#{pane_pid}"],
                              capture_output=True, text=True).stdout.strip()
        subprocess.run(["pkill", "-KILL", "-P", pane])
        time.sleep(0.5)

    def test_stop_without_monitor_still_clears_saved_state(self):
        self.leave_stale_state()
        (self.dir / "state" / "w1.123.exit").write_text("0\n")  # 遗留的退出记录也要清掉
        r = self.cli("stop", "w1", "--state-dir", str(self.dir / "state"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("No monitor is running", r.stdout + r.stderr)
        self.assertIn("cleared", r.stdout + r.stderr)
        self.assertFalse((self.dir / "state" / "w1.json").exists())
        self.assertEqual(list((self.dir / "state").glob("w1.*.exit*")), [])
        r = self.cli("once", "w1", *self.common)  # 全新的监控：直接发命令，而不是接着旧任务
        self.assertIn("start: sleep 60", r.stdout)

    def test_stop_all_without_monitor_clears_all_saved_states(self):
        self.leave_stale_state()
        (self.dir / "state" / "other.json").write_text("{}")
        r = self.cli("stop", "--all", "--state-dir", str(self.dir / "state"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(list((self.dir / "state").glob("*.json")), [])

    def test_stop_leaves_other_windows_state_alone(self):
        self.leave_stale_state()
        r = self.cli("stop", "other", "--state-dir", str(self.dir / "state"))
        self.assertNotEqual(r.returncode, 0)  # 没有监控也没有状态可清
        self.assertTrue((self.dir / "state" / "w1.json").exists())

    def test_run_unknown_window_in_task_file(self):
        r = self.cli("run", "nope", *self.common)
        self.assertEqual(r.returncode, 2)
        self.assertIn("nope", r.stdout + r.stderr)

    def test_run_missing_tmux_window_exits_1(self):
        (self.dir / "task.md").write_text("# tmux: ghost\n## task-list\necho hi\n")
        r = self.cli("run", "ghost", *self.common)
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not exist", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
