# E1执行修订 T7.1

当前进度和实际结果只见[PROGRESS](PROGRESS.md)。本次依据用户2026-09-30的决定，
落实[发布任务卡](MINIMAL_RELEASE_CHECKLIST.md)T7.1；不替换[T6协议](EXPERIMENT_FREEZE.md)
或[T7说明](E1_EXECUTION.md)。[修订登记](../experiments/t71_20260930/amendment.json)及
[执行封印](../experiments/t71_20260930/execution.json)记录本版边界。

## 三项前瞻修订

1. 裁判阶段上限由0.3改为0.5美元；E1阶段0.5美元、累计2美元停下报告、总5美元不变。
   原预算记录保留。裁判仍在取得全部实际待审文本后预留完整库存费用，不能足额预留就停止。
2. 以完整实际payload规范JSON的UTF-8字节数＋8192模板预留，作为事前输入token规划上界。
   “每token至少覆盖一个字节”是本次接受的估算假设，Claude分词器和服务商模板未公开核实，
   **不是实测token或供应商账单保证**。每次回执的`usage.prompt_tokens`超上界就记账停发；
   输出超`max_tokens`也停，不用后续较小请求抵消。无法取得usage或费用仍停止并保留未知。
3. `parallel_tool_calls=False`照T6原字段发送；只将其端点支持证据不足登记为可接受的未知。
   回执须恰好一个原生`aria_turn_decision`，否则记账停发。其余参数支持、价格、模型、
   供应商及禁回退门均保留，不删除实际请求字段，不通过试调用寻找可用模型。

题目、204位置与请求SHA、raw204/delivered204/配对27、盲评提示、40条人评抽样、
所有统计门与停止后不重试的规则均不变。E2当前允许调用数仍为0。

## 原封印与文档差异

任务05在`docs/VALIDATION.md`增加了用户批准裁判0.5美元的一句话；该文件在T6封印中，
所以旧验证器对当前整树会如实报文档字节不符。不能把这说成旧验证器原样通过。

本版保留最新文档，将已校验旧源码ZIP中的原文存为
[validation.t6.md](../experiments/t71_20260930/validation.t6.md)。新入口逐份复验T6的原始
104文件SHA（这一份使用原文引用），另核当前文档已登记的SHA；94份运行源码、原manifest、
原204请求均不变。T7的10份文件继续用原验证逻辑复验。T6/T7原seal不重写，新增改动另封。

## 运行

```powershell
.\.venv\Scripts\python.exe scripts/run_e1_t71.py
.\.venv\Scripts\python.exe scripts/run_e1_t71.py --online-preflight
.\.venv\Scripts\python.exe scripts/run_e1_t71.py --execute
.\.venv\Scripts\python.exe scripts/run_e1_t71.py --stage judge --execute
```

默认是离线校验，不读凭据、不打开账本、不联网。`ready_to_send=true`表示本版E1规划门通过，
实际发送还需当前公开端点复核、凭据可用及持久总账；不表示裁判预算或效果门已经通过。
`--online-preflight`只做不带凭据的公开GET，不访问账户API。

付费入口默认只取进程中的`OPENROUTER_API_KEY`；不要求用户把Key发到聊天。
仅取得用户明确同意后，才可使用`--allow-local-env`让执行程序提取本机`.env`里的这一个Key，
值不输出、不写日志、不纳入发布包。费用授权和读取该文件的边界分别遵守。

私有campaign路径仍为仓库同级`aria-private-experiments/t6_20260930_v1/`，不换run ID清账。
T7旧campaign若已存在或seal不符即停止待核，不能靠另开目录重跑。先持久claim再发送、
保存可见回执后记账、未知结果/费用停发均复用原T7实现。新策略只在独立函数作用域内绑定，
不改原模块全局变量，也不把原0.3美元变成0.5美元。

一次E1尝试结束后，无论是否完整，都保留204位置。只有完整且回执逐条核验通过才准备裁判。
裁判预算不足时保存全量计划与40条空人评表，不缩批次、不换种子，也不把缺失填0。
人工未完成仍只允许称Claude判断，不自动更改产品默认策略。

## 验证范围

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_t71_policy.py tests/test_e1_t71.py -q
```

替身检查规划上界与实际usage、多工具结果停发、完整位置/原请求指纹、未知费和持久账本，
并复验原T7预算未被污染。测试通过不能替代实跑；旧T7集成测试若直接调用旧T6整树验证，
仍会发现任务05的文档差异，不能修改旧测试来掩盖。当前统计和日志位置在PROGRESS登记。
