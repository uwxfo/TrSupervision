# tmux_supervisor

[English](README.en.md) | 中文

监督**一个** tmux 窗口里的训练任务：失败自动重试，成功后自动执行 `task.md` 中该窗口队列的下一条命令。仅用 Python 标准库。
多个窗口 = 多开几个监控进程（例如在 sv 会话里各开一个 tmux window）。

```bash
./tmuxsup run 1 --interval 600     # 常驻监督窗口 1（tmux 会话名）；每 600s 全量检查，任务结束约 2s 内立即处理
./tmuxsup once 1                   # 只检查一轮
./tmuxsup status 1                 # 监控进程是否在运行、当前状态、队列/worked/failed 条数
./tmuxsup stop 1                   # 结束窗口 1 的监控进程（不影响 tmux 里的训练）
./tmuxsup stop --all               # 结束全部监控进程
```
选项（run / once）：`--max-retries`（单条命令最大重试次数，默认 3）、`--retry-delay`（重试间隔秒，默认 10）、
`--no-auto-resume`、`--task-file`（默认本目录 `task.md`）、`--state-dir`、`--log-dir`、`--socket`（tmux -L）。
`run` 另有 `--interval`（秒，默认 1800）和 `--poll`（检测任务结束的轮询间隔，默认 2）。

## task.md（所有窗口的队列都在这一个文件里，运行中可随时编辑，每轮检查都会重新读取）
每个 `# tmux: <名字>` 一节，`<名字>` 为 tmux 会话名（或 `会话:窗口`），须与 tmux 实际名称完全一致：
```
# tmux: 0
workdir: /home/yluo/RLIPVISTR       # 命令的默认执行目录
## task-list
bash train_a.sh --batch_size 8
[workdir=/home/yluo/RLIPVISTR-1] bash train_b.sh   # 只对这一条换工作目录
## worked list                       # 成功结束的命令（程序移入，行尾带 `# [done log=... output_dir=...]`）
## failed list                       # 失败的命令，等你人工处理（程序移入，行尾带原因）
```
- **以 `#` 开头的整行是注释，不会执行**；想启用被注释的命令要删掉行首的 `# `。
- **运行中的命令一直留在 `task-list` 首行，成功后才移到 `worked list`**；重试用完或 shell 语法错误才移到 `failed list`。
  手动中断的任务也移到 `failed list`（原因 `manually interrupted ...`），然后继续下一条，两步都在等 30 秒之后（见下）。
- **工作目录**：`[workdir=路径]` 只对这一条生效（相对路径相对窗口的 `workdir`），**后面的任务不受影响**，仍用窗口的 `workdir`。
- **worked 条目的注释**：`bash a.sh  # [done log=/path/logs/1-123.log output_dir=/path/output/run_0425]`。
  `log` 是这次成功运行的完整输出日志；`output_dir` 是训练程序实际写 checkpoint 的目录（从日志里的 `output_dir='...'` 解析，没有则省略）。
- **failed 条目的注释**：`bash a.sh  # [failed exceeded max_retries=3 (exit code 1)]`（原因为英文）。
- 只有 `# [done ...]` / `# [failed ...]` 这种固定格式的尾巴会在读取任务时被剥掉，所以把条目移回 `task-list` 就能直接重跑；你自己写的其他 `# 注释` 不受影响。
- **终端输出和 failed 原因均为英文**；README、代码注释为中文。

## 判断成败的机制：退出码
tmux 本身不提供命令的退出码，所以工具启动命令时在外面包了一层，大致是：
```
( cd 工作目录 && bash -c '你的命令'; rc=$?; echo $rc > state/<窗口>.<run_id>.exit ) 2>&1 | tee -i -a logs/<窗口>-<run_id>.log
```
`rc=$?` 是命令的真实退出码，写入 `.exit` 文件（先写临时文件再 mv，保证内容完整）。工具据此判断，不解析屏幕输出：
- 退出码 0 → 成功，移入 worked list，启动队列下一条。
- 其他非 0 → 失败，进入重试（见下）。
- 130 / 143、退出码非 0 且日志末尾带 `KeyboardInterrupt` 或 torchrun 的 `got signal: 2/15`（torchrun 捕获 Ctrl-C 后以 1 退出）、或窗口已空闲但没有退出记录 → 视为**手动中断**：移入 `failed list`，继续下一条（不重试）。
- **手动中断不会立刻处理**：监控先打印提示并等待 **30 秒**（`supervisor.py` 里的 `INTERRUPT_WAIT`），之后才把任务移入 `failed list` 并启动下一条。
  这 30 秒里任务仍在 `task-list` 首行、不会启动任何新命令，你可以 `./tmuxsup stop <窗口>`（队列原样保留，被中断的任务不会移入 `failed list`），
  或对监控按 Ctrl-C 保住队列。Ctrl-C 后再 `run`，会发现任务已不在运行而再次判为中断，重新等 30 秒。普通失败（重试、OOM）不等待。
- 其余情况（包括 120，如磁盘写满）按普通错误重试。
- `tee -i` 让 tee 忽略 Ctrl-C。否则 Ctrl-C 会连 tee 一起杀掉，任务最后一次 flush 失败、Python 以 120 退出，真实退出码和最后的日志都会丢失。

## 重试规则
- 日志末尾含 CUDA OOM → 把 `batch_size` 减半后重试（最小 1，逐次累计：8→4→2→1）：
  - 命令里已有 `--batch_size N`（或 `--batch_size=N`），如 `bash a.sh --batch_size 8` → 直接减半这个数；
  - `batch_size` 只写在脚本里（如 `BATCH_SIZE=${BATCH_SIZE:-4}`）→ 从日志开头 `print(args)` 里读出实际生效的 `batch_size=N`，
    减半后在命令末尾**追加 `--batch_size <新值>`**（argparse 取最后一个）。
  - 脚本需把额外参数（`"$@"`）透传给训练程序，且放在自己的 `--batch_size` 之后。若日志里的 `batch_size` 与命令里的 `--batch_size`
    不一致，说明覆盖没生效，直接移入 `failed list` 并写明原因，不会白白重试。
  - **梯度累积**：每次 `batch_size` 减半，`accum_steps` 同时加倍，使逻辑 batch（`world_size * batch_size * accum_steps`）不变。
    旧 `accum_steps` 的读法同 `batch_size`：命令里的 `--accum_steps N`，没有则读日志 `print(args)` 里的 `accum_steps=N`，再没有按 1；
    新值在命令里原位改写，没有就**追加 `--accum_steps <新值>`**。8/1 → 4/2 → 2/4。`batch_size` 为奇数时减半取整、`accum_steps` 向上取整
    （5/1 → 2/3，逻辑 batch 由 5 变 6）。训练程序需支持 `--accum_steps`。
- 其余错误（读写错误等）→ 原命令重试。
- 每次重试前等待 `--retry-delay` 秒。超过 `--max-retries` 次仍失败，或 OOM 时找不到可再减小的 `--batch_size`，
  把命令移入 `failed list`，在终端打印原因（含日志最后 3 行），**窗口继续跑队列下一条**。
- 队列首条命令有 shell 语法错误 → 同样移入 `failed list`（原因：shell 语法错误），继续下一条。
- **自动续训**：重试时（不含队列里的新任务）先从上次运行的日志开头找 `output_dir='...'`（训练程序 `print(args)` 的输出，相对任务的工作目录），
  在该目录里选最新的完好 checkpoint（`checkpoint.pth` 与 `checkpointNNNN.pth` 中 `epoch` 最大者，不再无条件优先可能是旧运行遗留的 `checkpoint.pth`；`checkpoint_best.pth` 不算。
  `epoch` 从 checkpoint 里很小的 `data.pkl` 读出，不执行其中任何代码、不加载张量、不依赖 torch；相同则取修改时间新的。
  读不出 `epoch` 时：编号文件取文件名里的数字（按数值而非字符串比较），只要有候选读不出就全部退回按修改时间比较），
  给本次重试的命令加 `--resume <路径>`；命令里已有 `--resume` 则替换。找不到 `output_dir` 或没有可用 checkpoint
  （例如启动阶段就失败）时原命令重试；写到一半被打断的损坏文件会被跳过。`--no-auto-resume` 可关闭。
  要求脚本把额外参数透传给 `main.py`，且后出现的 `--resume` 覆盖脚本内默认值（argparse 取最后一个）。
- OOM 只匹配 GPU 显存不足：`OutOfMemoryError`（如 `torch.cuda.OutOfMemoryError`）、同一行含 `cuda`/`hip` 与 `out of memory`
  （`CUDA out of memory`、`CUDA error: out of memory`、`CUDA driver error`、NCCL 转述的 cuda 错误）、`CUDA_ERROR_OUT_OF_MEMORY`、
  `cudaErrorMemoryAllocation`、`Tried to allocate ... (GPU n ...)`、`CU*_STATUS_ALLOC_FAILED`。"illegal memory access"、
  CPU 内存不足（`DefaultCPUAllocator`、`Killed`）不算，减 batch 没用。

## 队列跑完后监控自己退出
一条命令成功（或最终失败）后，`task-list` 里没有剩余任务时，监控打印
`all tasks finished (worked list: N, failed list: M); monitor exits` 并**以退出码 0 退出**。failed list 里有条目不会改变退出码，请看打印的计数。
`run` 启动时 `task-list` 已为空则立即退出。
**所以想跑的任务请在最后一条结束前都加进 `task-list`**；监控退出后要继续就重新 `run`。
窗口里有非本工具启动的命令在运行时，监控会等待，不会退出。

## 自动结束监控（没有 paused 状态，也没有 resume 命令）
下面这些情况无法继续，监控进程会**在终端打印原因并以退出码 1 结束**，任务仍留在 `task-list` 首行。
手动中断（Ctrl-C、被杀、终端被关）**不在此列**：等 30 秒后任务移入 `failed list` 再继续下一条；若你想停掉整个队列，用 `./tmuxsup stop <窗口>`（30 秒内任务仍留在 `task-list`）。
手动中断里，"窗口空闲但没有退出记录"由轮询（`--poll`）发现，启动后 15 秒内为宽限期，不必等 `--interval`。
- tmux 窗口不存在或运行中被关闭；`task.md` 里已没有该窗口的队列。
- tmux 无响应：每次调用 `tmux` / `pgrep` 都有 15 秒超时，不会再因为某次调用卡住而无声冻结；超时则以 `tmux did not respond` 为原因结束监控。

处理完问题后重新 `./tmuxsup run <窗口>`，会从 `task-list` 首行重新开始。`stop` 或 Ctrl-C 结束监控时退出码为 0。

## Ctrl-C 与 `stop` 的区别，以及再次 `run` 的行为
- **对监控进程按 Ctrl-C** 只是暂离，状态会保留。再次 `run` 会接着监控窗口里仍在跑的任务（期间已结束则正常处理结果），
  不会重复启动，也不会当成外部命令。监控进程被杀、终端被关闭同理。
- **`stop`**、队列跑空、自动结束这几种情况会清空状态，下次 `run` 是全新的监控，因此：
- 窗口里正在跑的命令（包括上次监控启动、仍在运行的）一律当作**外部命令**：在终端打印提示，等它结束，**不重试**。
- **提示符处的残留字符**（没发出的半行、续行提示符 `>`、Ctrl-R 搜索）会和工具键入的命令拼在一起导致报错。所以发命令前，工具会对空闲的 shell
  先发一个 `C-c` 清掉输入行；窗口里有命令在跑时一个键都不发，外部程序不会被打断；若在"判定空闲"到"发键"之间恰好有程序启动，
  任务留在 `task-list`，等它结束后再启动。
- **注意重复运行**：若你在任务 T 运行中 `stop` 了监控，T 还在 `task-list` 首行，T 结束后新监控会再启动一次 T。
  `stop` 前后请自行处理队列（例如把 T 删掉或移到 worked list）。
- 同一窗口同时只允许一个监控进程（文件锁），重复 `run` 会被拒绝并提示 pid。

## 监督周期
- `--interval` 单位为**秒**，默认 1800，是全量检查周期；工具本身不调用模型，不消耗 token，周期只影响 CPU，建议 600 左右，不必低于 60。
- 工具启动的任务一结束，约 2 秒内（`--poll`）就会立即处理，不必等满周期。周期只兜底这些情况：手动改了 task.md（新增命令最多等一个周期才启动）、tmux 窗口状态变化等。

## 注意事项
- 窗口 shell 需空闲才会启动命令。命令经 `bash -c` 执行，shell 别名/函数不可用，conda 环境变量可继承。
- 日志在 `logs/`，事件在 `state/events.log`。

测试：`python3 -m unittest discover -s tests -t .`
