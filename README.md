# tmux_supervisor

监督 tmux 窗口里的训练任务：失败自动重试，成功后自动执行 `task.md` 中该窗口队列的下一条命令。仅用 Python 标准库。

```bash
./tmuxsup run --interval 600              # 常驻；每 600s 全量检查，任务结束会立即处理（建议放在单独的 tmux 窗口）
./tmuxsup run --windows 0 sv --interval 60  # 只监督窗口 0 和 sv
./tmuxsup once | status | resume 0        # 检查一轮 / 查看状态 / 恢复被暂停的窗口
```
公共选项：`--max-retries`（单条命令最大重试次数，默认 3）、`--task-file`（默认本目录 `task.md`）、`--state-dir`、`--log-dir`、`--socket`（tmux -L）。

## task.md（所有窗口的队列都在这一个文件里，运行中可随时编辑，每轮检查都会重新读取）
每个 `# tmux: <名字>` 一节，`<名字>` 为 tmux 会话名（或 `会话:窗口`），须与 tmux 实际名称完全一致：
```
# tmux: 0
workdir: /home/yluo/RLIPVISTR      # 命令的执行目录
## task-list
bash train_detant_vidhoi.sh --batch_size 8
## worked list
```
- `--windows` 只能选 task.md 里已有的窗口；要监督新窗口，先在 task.md 里加对应一节（队列可为空）。
- **以 `#` 开头的行是注释，不会执行**。想启用被注释的命令，要删掉行首的 `# `。
- 命令启动时自动从 `## task-list` 移到 `## worked list`。

## 判断成败的机制：退出码
tmux 本身不提供命令的退出码，所以工具启动命令时在外面包了一层，大致是：
```
( cd 工作目录 && bash -c '你的命令'; rc=$?; echo $rc > state/<窗口>.<run_id>.exit ) 2>&1 | tee -a logs/<窗口>-<run_id>.log
```
`rc=$?` 是命令的真实退出码，写入 `.exit` 文件（先写临时文件再 mv，保证内容完整）。工具据此判断，不解析屏幕输出：
- 退出码 0 → 启动队列下一条。
- 130 / 143（Ctrl-C / 终止）→ 视为人工中断，暂停，需 `resume`。
- 其他非 0 → 失败，进入重试（见下）。
- 没有 `.exit` 文件但窗口已空闲 → 进程被杀或终端被关，暂停。

## 重试规则
- 日志末尾含 CUDA OOM → 把命令里的 `--batch_size N`（或 `--batch_size=N`）减半后重试（最小 1，逐次累计：8→4→2→1），例如 `bash a.sh --batch_size 8`；脚本需把额外参数透传给训练程序。
- 其余错误（读写错误等）→ 原命令重试。
- 重试超过 `--max-retries`，或 OOM 时命令里没有 `--batch_size` / 已为 1，则暂停。
- **自动续训**：重试时（不含队列里的新任务）先从上次运行的日志开头找 `output_dir='...'`（训练程序 `print(args)` 的输出，相对任务的 `workdir`），在该目录里选最新的完好 checkpoint（`checkpoint.pth`，其次编号最大的 `checkpointNNNN.pth`，`checkpoint_best.pth` 不算），给本次重试的命令加 `--resume <路径>`；命令里已有 `--resume` 则替换。
  找不到 `output_dir` 或没有可用 checkpoint 时（例如启动阶段就失败）原命令重试；写到一半被打断的损坏文件会被跳过。`--no-auto-resume` 可关闭。要求脚本把额外参数透传给 `main.py`，且 `--resume` 后出现的值覆盖脚本内默认值（argparse 取最后一个）。
- OOM 只匹配 CUDA out of memory、CUBLAS/CUDNN ALLOC_FAILED；"illegal memory access" 不算，减 batch 没用。

## 监督周期
- `--interval` 单位为**秒**，默认 1800，是全量检查周期；工具本身不调用模型，不消耗 token，周期只影响 CPU，建议 600 左右，不必低于 60。
- 工具启动的任务一结束，约 2 秒内（`--poll`）就会立即处理，不必等满周期。周期只兜底这些情况：被异常杀掉没有退出记录、手动改了 task.md（新增命令最多等一个周期才启动）、`resume` 之后、tmux 窗口后来才建好。

## 注意事项
- **只会重试/接管本工具启动的命令。** 在窗口里手动敲的命令没有被包装，拿不到退出码，窗口状态显示 `external`，工具只等它结束，不会重试；要被监督就写进 task.md 队列。
- 窗口 shell 需空闲才会启动命令。命令经 `bash -c` 执行，shell 别名/函数不可用，conda 环境变量可继承。
- 队列首条命令有 shell 语法错误 → 暂停，不执行。
- 日志在 `logs/`，事件在 `state/events.log`，各窗口状态用 `./tmuxsup status` 查看。

测试：`python3 -m unittest discover -s tests -t .`
