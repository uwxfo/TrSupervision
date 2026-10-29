# tmux_supervisor

English | [中文](README_ZH.md)

Supervises the training tasks running in **one** tmux window: a failed command is retried automatically, and a
successful one is followed by the next command from that window's queue in `task.md`. Python standard library only.
To supervise several windows, run several monitor processes (for example, one tmux window per monitor inside a `sv` session).

```bash
./tmuxsup run 1 (--interval 1800 --max_retries 3)    # supervise window 1 (a tmux session name); full check every 600 s, a finished task is handled within ~2 s
./tmuxsup once 1                   # do a single check
./tmuxsup status 1                 # is a monitor running, current state, queue / worked / failed counts
./tmuxsup stop 1                   # end the monitor of window 1 (the training in tmux is not touched)
./tmuxsup stop --all               # end all monitors
```
Options (run / once / status): `--max-retries` (max retries per command, default 3), `--retry-delay` (seconds between retries, default 10),
`--no-auto-resume`, `--task-file` (default: `task.md` in this folder), `--state-dir`, `--log-dir`, `--socket` (tmux -L).
`run` also takes `--interval` (seconds, default 1800) and `--poll` (polling interval for detecting a finished task, default 2).

## task.md (the queues of all windows live in this one file; edit it any time, it is re-read on every check)
One section per `# tmux: <name>`, where `<name>` is the tmux session name (or `session:window`) and must match the real tmux name exactly:
```
# tmux: 0
workdir: /home/yluo/RLIPVISTR       # default working directory of the commands
## task-list
bash train_a.sh --batch_size 8
[workdir=/home/yluo/RLIPVISTR-1] bash train_b.sh   # change the working directory for this one command only
## worked list                       # commands that finished successfully (moved here by the tool, with `# [done log=... output_dir=...]`)
## failed list                       # commands that failed, waiting for you (moved here by the tool, with the reason at the end of the line)
```
- **A whole line starting with `#` is a comment and is not executed.** To enable a commented-out command, remove the leading `# `.
- **A running command stays on the first line of `task-list` until it succeeds**, then moves to `worked list`. It moves to
  `failed list` only after retries are exhausted or on a shell syntax error. A manually interrupted task also moves to `failed list` (reason `manually interrupted ...`) and the next command starts, both after a 30 s wait (see below).
- **Working directory**: `[workdir=path]` applies to that one command only (a relative path is relative to the window's `workdir`);
  **the following tasks are not affected** and keep using the window's `workdir`.
- **Note on a worked entry**: `bash a.sh  # [done log=/path/logs/1-123.log output_dir=/path/output/run_0425]`.
  `log` is the full output log of the successful run; `output_dir` is the directory where the training program writes its checkpoints
  (parsed from `output_dir='...'` in the log; omitted when absent).
- **Note on a failed entry**: `bash a.sh  # [failed exceeded max_retries=3 (exit code 1)]`.
- Only a trailing `# [done ...]` / `# [failed ...]` in exactly this format is stripped when a task is read, so you can move an entry
  back into `task-list` and it runs as is. Any other `# comment` you write is left alone.
- **All terminal output and failure reasons are in English**; the Chinese README and the code comments are in Chinese.

## How success or failure is decided: the exit code
tmux does not expose the exit code of a command, so the tool wraps every command it starts, roughly like this:
```
( cd <workdir> && bash -c '<your command>'; rc=$?; echo $rc > state/<window>.<run_id>.exit ) 2>&1 | tee -i -a logs/<window>-<run_id>.log
```
`rc=$?` is the real exit code of the command; it is written to the `.exit` file (to a temp file first, then `mv`, so the content is always complete).
The tool decides from that file and never parses the screen:
- Exit code 0 -> success: the command moves to the worked list and the next queued command starts.
- Any other non-zero code -> failure: retry (see below).
- 130 / 143, a non-zero exit whose log tail contains `KeyboardInterrupt` or torchrun's `got signal: 2/15` (torchrun catches Ctrl-C and exits with 1), or the window is idle with no exit record -> a **manual interruption**: moved to `failed list`, the next command starts (no retry).
- **A manual interruption is not acted on at once**: the monitor prints a notice and waits **30 s** (`INTERRUPT_WAIT` in `supervisor.py`) before it moves the
  task to `failed list` and starts the next command. During these 30 s the task is still on the first line of `task-list` and nothing new starts, so you
  can `./tmuxsup stop <window>` (the queue stays exactly as it was; the interrupted task is not moved to `failed list`) or Ctrl-C the monitor to keep the queue.
  After a Ctrl-C the next `run` finds the task not running and treats it as interrupted again, with another 30 s wait. Ordinary failures (retries, OOM) do not wait.
- Anything else (including 120, e.g. a full disk) is an ordinary error and is retried.
- `tee -i` makes `tee` ignore Ctrl-C. Without it, Ctrl-C also kills `tee`, the task's last flush fails and Python exits with 120, losing the real exit code and the final log lines.

## Retry rules
- If the end of the log contains a CUDA OOM, `batch_size` is halved and the command is retried (minimum 1, cumulative: 8 -> 4 -> 2 -> 1):
  - the command already has `--batch_size N` (or `--batch_size=N`), e.g. `bash a.sh --batch_size 8` -> that number is halved;
  - `batch_size` is only set inside the script (e.g. `BATCH_SIZE=${BATCH_SIZE:-4}`) -> the tool reads the effective `batch_size=N` from the
    `print(args)` line at the top of the log, halves it and **appends `--batch_size <new>`** to the command (argparse keeps the last one).
  - The script must pass extra arguments (`"$@"`) through to the training program, after its own `--batch_size`. If the log's `batch_size`
    differs from the command's `--batch_size`, the override is not applied; the command moves to `failed list` with that reason instead of
    retrying in vain.
  - **Gradient accumulation**: whenever `batch_size` is halved, `accum_steps` is doubled so the logical batch size
    (`world_size * batch_size * accum_steps`) stays the same. The old `accum_steps` is read like `batch_size`: `--accum_steps N` in the command,
    else `accum_steps=N` from the log's `print(args)`, else 1; the new value is rewritten in place or **appended as `--accum_steps <new>`**.
    8/1 -> 4/2 -> 2/4. An odd `batch_size` is halved with integer division and `accum_steps` is rounded up (5/1 -> 2/3, logical batch 5 -> 6).
    The training program must support `--accum_steps`.
- **Port in use** (`EADDRINUSE` / `address already in use` / `errno 98` at the end of the log, e.g. torchrun's `DistNetworkError ... port: 29501 ... EADDRINUSE`):
  the retry runs the command with a fresh `MASTER_PORT`, and the command itself asks the system for a free port when it starts
  (`$(python3 -c '... bind(("", 0)) ...')`), so it is correct on the machine that really runs the training, also inside `srun` / `ssh` nested shells.
  An existing `MASTER_PORT=<digits>` in the command is rewritten in place, otherwise `export MASTER_PORT=...;` is put in front. The script must read the
  `MASTER_PORT` environment variable (`${MASTER_PORT:-default}`) and `python3` must be on the PATH of the window shell. Every retry gets a new port. The
  `task.md` entry is not changed. Port retries count toward `--max-retries`. If the log reports the **same port** again after the switch (the script
  hard-codes `--master_port`, so the override has no effect), the command moves to `failed list` at once with that reason instead of retrying in vain.
- Any other error (read/write errors, ...) -> retry the command unchanged.
- The tool waits `--retry-delay` seconds before each retry. If it still fails after `--max-retries` retries, or on an OOM when there is no
  `batch_size` left to reduce, the command moves to `failed list`, the reason is printed to the terminal (with the last 3 log lines),
  and **the window carries on with the next queued command**.
- A shell syntax error in the first queued command -> it also moves to `failed list` (reason: shell syntax error) and the next command starts.
- **Auto-resume**: on a retry (not for a new task from the queue), the tool finds `output_dir='...'` at the start of the previous run's log
  (printed by the training program via `print(args)`, relative to the task's working directory) and picks the newest intact checkpoint in that directory
  (the one with the largest `epoch` among `checkpoint.pth` and `checkpointNNNN.pth`, so a stale `checkpoint.pth` is not preferred blindly;
  `checkpoint_best.pth` is not training progress and is ignored). The `epoch` is read from the checkpoint's small `data.pkl` without executing
  any code or loading the tensors (and without torch); on a tie the newer modification time wins. If an `epoch` cannot be read (for numbered files
  it is taken from the file name, compared numerically), all candidates are compared by modification time instead.
  It then adds `--resume <path>` to the command for this retry; an existing `--resume` is replaced. If there is no `output_dir` or no usable
  checkpoint (for example the run failed during startup), the command is retried unchanged. Half-written, corrupt files are skipped.
  `--no-auto-resume` turns it off. The script has to pass extra arguments through to `main.py`, and a later `--resume` has to override the
  script's default (argparse keeps the last one).
- Only GPU out-of-memory counts as OOM: `OutOfMemoryError` (e.g. `torch.cuda.OutOfMemoryError`), a `cuda`/`hip` line with `out of memory`
  (`CUDA out of memory`, `CUDA error: out of memory`, `CUDA driver error`, NCCL relaying one), `CUDA_ERROR_OUT_OF_MEMORY`, `cudaErrorMemoryAllocation`,
  `Tried to allocate ... (GPU n ...)`, `CU*_STATUS_ALLOC_FAILED`. "illegal memory access" and CPU out-of-memory (`DefaultCPUAllocator`, `Killed`)
  do not count, because a smaller batch would not help.

## When the whole queue is done, the monitor exits
When a command succeeds (or fails for good) and `task-list` has nothing left, the monitor prints
`all tasks finished (worked list: N, failed list: M); monitor exits` and **exits with code 0**. Entries in the failed list do not change the exit code,
so check the printed counts. If `task-list` is already empty at `run`, it exits immediately.
**So add every task you want to run to `task-list` before the last one finishes**; once the monitor has exited, start it again with `run`.
While a command that the tool did not start is still running in the window, the monitor waits and does not exit.

## Automatic end of monitoring (there is no paused state and no resume command)
In the following cases it cannot go on, so the monitor **prints the reason to the terminal and exits with code 1**.
The task stays on the first line of `task-list`.
A manual interruption (Ctrl-C, process killed, terminal closed) is **not** one of them: after a 30 s wait the task moves to `failed list` and the next command starts;
to stop the whole queue use `./tmuxsup stop <window>` (within the 30 s the task stays in `task-list`). For the "idle with no exit record" kind of interruption, the monitor notices it in its
polling loop (`--poll`) after a 15 s start grace period, without waiting for `--interval`.
- The tmux window does not exist or was closed while running, or `task.md` no longer has a queue for that window.
- tmux does not respond: every `tmux` / `pgrep` call has a 15 s timeout, so a hung call can no longer freeze the monitor silently; on a timeout it exits with the reason `tmux did not respond`.

After you have dealt with the problem, run `./tmuxsup run <window>` again. It restarts from the first line of `task-list`.
Ending the monitor with `stop` or Ctrl-C exits with code 0.

## Ctrl-C vs `stop`, and what the next `run` does
- **Ctrl-C on the monitor** is a temporary leave: its state is kept. The next `run` picks up the task still running in the window (or handles its result
  if it finished meanwhile), so nothing is started twice and nothing is treated as an external command. A killed monitor or a closed terminal behaves the same.
- **`stop`**, the queue draining, and the automatic-end cases clear the state, so the next `run` is a fresh monitoring session. If no monitor is running
  any more (it was ended with Ctrl-C, killed, or its terminal closed), `stop` clears the saved state itself and prints `cleared saved state of: ...`
  (exit code 0; with nothing to clear it prints `No monitor is running` and exits 1), so a following `run` does not resume the old task:
- Any command already running in the window (including one started by the previous monitor that is still running) is treated as an **external command**:
  a notice is printed, the monitor waits for it to finish and **does not retry it**.
- **Leftover characters at the prompt** (an unsent half line, a `>` continuation prompt, a Ctrl-R search) would be glued to the command the tool types
  and break it. So right before typing a command the tool sends `C-c` to the idle shell to clear the line. It sends nothing while a command is running
  in the window, so an external program is never interrupted; if one starts between the idle check and the keystroke, the task stays in `task-list`
  and starts after it finishes.
- **Beware of duplicate runs**: if you `stop` the monitor while task T is running, T is still on the first line of `task-list`, and after T finishes the new
  monitor will start T once more. Deal with the queue yourself around `stop` (for example delete T or move it to the worked list).
- Only one monitor per window is allowed (file lock). A second `run` is refused and the pid of the running one is shown.

## Supervision interval
- `--interval` is in **seconds** (default 1800) and is the period of the full check. The tool never calls a model, so it uses no tokens; the interval only
  affects CPU. About 600 is a good value, there is no reason to go below 60.
- When a task started by the tool finishes, it is handled within about 2 s (`--poll`); there is no need to wait for the full interval. The interval is only the
  fallback for things like: you edited task.md (a newly added command starts after at most one interval), or the tmux window changed state.

## Notes
- The window shell must be idle for the tool to start a command. Commands run through `bash -c`, so shell aliases / functions are not available;
  conda environment variables are inherited.
- **Nested shells**: if the window runs `srun` / `salloc` / `ssh` (for example `srun --pty bash` into a compute node), the local machine cannot see the commands
  inside, so busy/idle is decided by the cursor line: it counts as idle only when it ends with a prompt character (`$ # % >`). Leftover text would make it
  look like a running program, so when the tool needs to start a command and the window looks busy, it sends `C-a C-k` (beginning of line + kill to end of line)
  **once**, waits 0.3 s and judges once more: if the line is clear the command starts, otherwise a program really is running and the tool waits as for any
  external command (no further keys are sent in that wait). `C-a C-k` is not a signal, so it never kills the program; at worst a few control characters
  reach its input. It is not sent when the queue is empty. A program whose output line happens to end with a prompt character is still misjudged as idle.
  A `>` continuation prompt (unclosed quote) cannot be cleared in a nested shell (`C-c` would be needed), unlike in a plain window.
- **Scrolling (copy-mode)**: if the window is in tmux copy-mode (scrolling through history), the tool leaves that mode before sending keys, so the command is not swallowed.
- Logs are in `logs/`, events in `state/events.log`.

Tests: `python3 -m unittest discover -s tests -t .`
