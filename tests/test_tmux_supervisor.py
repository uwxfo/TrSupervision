import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import zipfile

from tmux_supervisor.supervisor import Supervisor, find_checkpoint, with_resume
from tmux_supervisor.tasks import load_windows, move_first_task_to_worked
from tmux_supervisor.tmuxctl import Tmux

TASKS = """# tmux: a
workdir: /tmp

## task-list
# 注释
bash x.sh

bash y.sh --batch_size 2

## worked list
old

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
        self.assertEqual(w["a"].tasks, ["bash x.sh", "bash y.sh --batch_size 2"])
        self.assertEqual(w["a"].workdir, "/tmp")

    def test_move(self):
        self.assertFalse(move_first_task_to_worked(self.f, "a", "bash nope"))
        self.assertTrue(move_first_task_to_worked(self.f, "a", "bash x.sh"))
        self.assertEqual(load_windows(self.f)["a"].tasks, ["bash y.sh --batch_size 2"])
        text = self.f.read_text()
        self.assertIn("old\nbash x.sh", text)
        self.assertEqual(load_windows(self.f)["b"].tasks, ["bash z.sh"])  # 其他窗口不受影响

    def test_move_creates_worked_list(self):
        self.assertTrue(move_first_task_to_worked(self.f, "b", "bash z.sh"))
        self.assertTrue(self.f.read_text().rstrip().endswith("## worked list\nbash z.sh"))


def make_ckpt(path: Path, valid: bool = True):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data.pkl", "x" * 1000)
    if not valid:  # 模拟写到一半被打断：砍掉文件尾的 zip 目录
        path.write_bytes(path.read_bytes()[:600])


class ResumeHelperTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_find_checkpoint_prefers_latest_valid(self):
        self.assertIsNone(find_checkpoint(self.d))
        make_ckpt(self.d / "checkpoint0019.pth")
        make_ckpt(self.d / "checkpoint0039.pth")
        make_ckpt(self.d / "checkpoint_best.pth")  # 不是训练进度，不应被选中
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0039.pth")
        make_ckpt(self.d / "checkpoint.pth")
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint.pth")

    def test_find_checkpoint_skips_truncated(self):
        make_ckpt(self.d / "checkpoint0019.pth")
        make_ckpt(self.d / "checkpoint.pth", valid=False)
        self.assertEqual(find_checkpoint(self.d).name, "checkpoint0019.pth")

    def test_with_resume(self):
        p = Path("/a b/ck.pth")
        self.assertEqual(with_resume("bash a.sh --batch_size 4", p), "bash a.sh --batch_size 4 --resume '/a b/ck.pth'")
        self.assertEqual(with_resume("bash a.sh --resume old.pth --seed 1", p), "bash a.sh --resume '/a b/ck.pth' --seed 1")
        self.assertEqual(with_resume("bash a.sh --resume=old.pth", p), "bash a.sh --resume '/a b/ck.pth'")


class TmuxIntegrationTest(unittest.TestCase):
    """用独立 socket 的真实 tmux server 跑端到端流程。"""

    def setUp(self):
        self.sock = f"tmuxsup-test-{os.getpid()}-{time.time_ns()}"  # 每个测试独立 server，避免 kill-server 后立即复用 socket 的竞态
        self.tm = Tmux(self.sock)
        self.dir = Path(tempfile.mkdtemp())
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-s", "w1", "-x", "200", "bash", "--norc"], check=True)
        self.addCleanup(lambda: subprocess.run(["tmux", "-L", self.sock, "kill-server"], capture_output=True))
        self.sup = Supervisor(self.dir / "task.md", self.dir / "state", self.dir / "logs", self.tm, echo=lambda s: None)

    def write_tasks(self, body, header="workdir: {d}\n"):
        (self.dir / "task.md").write_text(f"# tmux: w1\n{header.format(d=self.dir)}{body}")

    def drive(self, cond, timeout=40):
        end = time.time() + timeout
        while time.time() < end:
            self.sup.check("w1")
            if cond():
                return
            time.sleep(0.3)
        self.fail(f"超时；状态={self.sup.load_state('w1')}")

    def test_queue_runs_in_order(self):
        self.write_tasks("## task-list\necho A >> out\necho B >> out\n")
        self.drive(lambda: (self.dir / "out").exists() and (self.dir / "out").read_text() == "A\nB\n"
                   and self.sup.load_state("w1")["run_id"] is None)
        self.assertEqual(load_windows(self.dir / "task.md")["w1"].tasks, [])

    def test_oom_halves_batch_other_errors_keep_batch(self):
        script = self.dir / "t.sh"
        # 第一次：BATCH_SIZE>2 报 CUDA OOM；之后：第一次遇到 flag 缺失时报 IO 错误，再之后成功
        script.write_text(
            'BATCH_SIZE=$2\n'
            'echo "bs=$BATCH_SIZE" >> seen\n'
            'if [ "$BATCH_SIZE" -gt 2 ]; then echo "RuntimeError: CUDA out of memory." >&2; exit 1; fi\n'
            'if [ ! -f flag ]; then touch flag; echo "OSError: read failure" >&2; exit 1; fi\n'
            'echo ok >> seen\n')
        self.write_tasks(f"## task-list\nbash {script} --batch_size 8\n")
        self.drive(lambda: "ok" in (self.dir / "seen").read_text() if (self.dir / "seen").exists() else False)
        self.assertEqual((self.dir / "seen").read_text().split(), ["bs=8", "bs=4", "bs=2", "bs=2", "ok"])

    def test_pause_after_max_retries(self):
        self.sup.max_retries = 1
        self.write_tasks("## task-list\nbash -c 'echo x >> n; exit 3'\necho never >> n\n")
        self.drive(lambda: self.sup.load_state("w1")["status"] == "paused")
        self.assertEqual((self.dir / "n").read_text().split(), ["x", "x"])  # 首次 + 1 次重试，后续任务不执行
        self.sup.resume("w1")
        self.assertEqual(self.sup.load_state("w1")["status"], "idle")

    def test_oom_without_batch_arg_pauses(self):
        self.write_tasks("## task-list\nbash -c 'echo CUDA out of memory >&2; exit 1'\n")
        self.drive(lambda: self.sup.load_state("w1")["status"] == "paused")
        self.assertIn("--batch_size", self.sup.load_state("w1")["reason"])

    def _resume_script(self, with_ckpt: bool):
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
        self._resume_script(with_ckpt=True)
        self.drive(lambda: (self.dir / "seen").exists())
        self.assertEqual((self.dir / "seen").read_text().strip(), f"args: --resume {self.dir}/od/checkpoint.pth")

    def test_retry_without_checkpoint_reruns_as_is(self):
        self._resume_script(with_ckpt=False)
        self.drive(lambda: (self.dir / "seen").exists())
        self.assertEqual((self.dir / "seen").read_text().strip(), "args:")

    def test_no_auto_resume_flag(self):
        self.sup.auto_resume = False
        self._resume_script(with_ckpt=True)
        self.drive(lambda: (self.dir / "seen").exists())
        self.assertEqual((self.dir / "seen").read_text().strip(), "args:")

    def test_ctrl_c_pauses(self):
        self.write_tasks("## task-list\nsleep 60\n")
        self.drive(lambda: self.sup.load_state("w1")["status"] == "running")
        time.sleep(1)
        subprocess.run(["tmux", "-L", self.sock, "send-keys", "-t", "=w1:", "C-c"], check=True)
        self.drive(lambda: self.sup.load_state("w1")["status"] == "paused")

    def test_missing_session_does_not_crash(self):
        (self.dir / "task.md").write_text("# tmux: nope\n## task-list\necho hi\n")
        self.sup.check("nope")
        self.assertEqual(self.sup.load_state("nope")["status"], "no-session")


if __name__ == "__main__":
    unittest.main()
