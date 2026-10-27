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
  `failed list` only after retries are exhausted or on a shell syntax error. A manually interrupted task goes to neither list; it stays in `task-list`.
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
( cd <workdir> && bash -c '<your command>'; rc=$?; echo $rc > state/<window>.<run_id>.exit ) 2>&1 | tee -a logs/<window>-<run_id>.log
```
`rc=$?` is the real exit code of the command; it is written to the `.exit` file (to a temp file first, then `mv`, so the content is always complete).
The tool decides from that file and never parses the screen:
- Exit code 0 -> success: the command moves to the worked list and the next queued command starts.
- Any other non-zero code -> failure: retry (see below).
- 120 / 130 / 143, or the window is idle with no exit record -> treated as a **manual interruption**, see below.

## Retry rules
- If the end of the log contains a CUDA OOM, the `--batch_size N` (or `--batch_size=N`) in the command is halved and the command is retried
  (minimum 1, cumulative: 8 -> 4 -> 2 -> 1), e.g. `bash a.sh --batch_size 8`. The script must pass extra arguments through to the training program.
- Any other error (read/write errors, ...) -> retry the command unchanged.
- The tool waits `--retry-delay` seconds before each retry. If it still fails after `--max-retries` retries, or on an OOM when the command has no
  `--batch_size` left to reduce, the command moves to `failed list`, the reason is printed to the terminal (with the last 3 log lines),
  and **the window carries on with the next queued command**.
- A shell syntax error in the first queued command -> it also moves to `failed list` (reason: shell syntax error) and the next command starts.
- **Auto-resume**: on a retry (not for a new task from the queue), the tool finds `output_dir='...'` at the start of the previous run's log
  (printed by the training program via `print(args)`, relative to the task's working directory) and picks the newest intact checkpoint in that directory
  (`checkpoint.pth`, then the highest-numbered `checkpointNNNN.pth`; `checkpoint_best.pth` is not training progress and is ignored).
  It then adds `--resume <path>` to the command for this retry; an existing `--resume` is replaced. If there is no `output_dir` or no usable
  checkpoint (for example the run failed during startup), the command is retried unchanged. Half-written, corrupt files are skipped.
  `--no-auto-resume` turns it off. The script has to pass extra arguments through to `main.py`, and a later `--resume` has to override the
  script's default (argparse keeps the last one).
- Only CUDA out of memory and CUBLAS/CUDNN ALLOC_FAILED count as OOM; "illegal memory access" does not, because a smaller batch would not help.

## When the whole queue is done, the monitor exits
When a command succeeds (or fails for good) and `task-list` has nothing left, the monitor prints
`all tasks finished (worked list: N, failed list: M); monitor exits` and **exits with code 0**. Entries in the failed list do not change the exit code,
so check the printed counts. If `task-list` is already empty at `run`, it exits immediately.
**So add every task you want to run to `task-list` before the last one finishes**; once the monitor has exited, start it again with `run`.
While a command that the tool did not start is still running in the window, the monitor waits and does not exit.

## Automatic end of monitoring (there is no paused state and no resume command)
In the following cases the situation cannot be handled safely, so the monitor **prints the reason to the terminal and exits with code 1**.
The task stays on the first line of `task-list`:
- The task exited with 130 / 143 (Ctrl-C / terminate) or **120** (Python failed to flush its output: common when Ctrl-C passes through `tee`, but
  it can also happen when the disk is full).
- The window is idle but there is no exit record (the process was killed or the terminal was closed). There is no record to read in this case, so the
  monitor notices it in its polling loop (`--poll`), after a 15 s start grace period. It ends within about 15 s and does not wait for `--interval`.
- The tmux window does not exist or was closed while running, or `task.md` no longer has a queue for that window.
- tmux does not respond: every `tmux` / `pgrep` call has a 15 s timeout, so a hung call can no longer freeze the monitor silently; on a timeout it exits with the reason `tmux did not respond`.

After you have dealt with the problem, run `./tmuxsup run <window>` again. It restarts from the first line of `task-list`.
Ending the monitor with `stop` or Ctrl-C exits with code 0.

## Behaviour of a new `run`: a new monitoring session, no inherited state
`run` clears the window's previous state when it starts. Therefore:
- Any command already running in the window (including one started by the previous monitor that is still running) is treated as an **external command**:
  a notice is printed, the monitor waits for it to finish and **does not retry it**.
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
- Logs are in `logs/`, events in `state/events.log`.

Tests: `python3 -m unittest discover -s tests -t .`
