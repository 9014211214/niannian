# E1执行修复 T7.2：预检、错误记录与复跑条件

当前状态和任务结果见[PROGRESS](PROGRESS.md)。本版是独立修复，继承
[T6](EXPERIMENT_FREEZE.md)、[T7](E1_EXECUTION.md)与[T7.1](E1_EXECUTION_T71.md)原件，
不覆盖旧请求、失败回执或费用。[修订登记](../experiments/t72_20260930/amendment.json)
与[修复封印](../experiments/t72_20260930/execution.json)绑定本版源码及测试。

## 修复内容

T7.1在预检副本中删除`parallel_tool_calls`，实际发送仍带`false`，且
`require_parameters=true`。[官方路由规则](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters)
要求供应商支持请求参数；响应后只有一个工具调用，不能证明请求能通过发送前的路由筛选。
本版核验完整实际payload，`false`也需要支持证据，并将规范JSON的SHA与预检结果绑定。
字段、嵌套内容或供应商设置改变后，原检查结果失效。

离线检查覆盖204位置的全部实际请求，复用原模型、供应商、价格、token限制检查；
支持字段缺失明确阻断。公开端点复核直接检查实际payload，不删字段或放宽
`require_parameters`、隐私约束、模型及供应商锁定。公开元数据通过仍不是账户路由或费用保证。

独立传输组件继续单次发送、禁止自动重试/重定向/环境代理，保留HTTP状态、固定类别、
严格格式的请求标识及有界数值Retry-After。错误正文最多读取64 KiB，并受同一绝对期限约束；
无法读取、非JSON、超长或截断时仍保留已取得的HTTP状态。HTTP类别只描述状态，
例如404表示not_found，不能自动认定为某个参数不支持。响应头的X-Generation-Id记为
`header_generation_id`，正文顶层id记为`generation_id`；两者不同时显式标记冲突，
不互相覆盖。HTTP200读取/解析/语义校验失败也保留安全请求标识，便于查账。

不保存完整错误消息、响应正文、任意响应头、隐藏推理或凭据。成功HTTP200仍走原模型/
usage/可见内容校验；补充JSON转义后的凭据回显拦截。未来使用本组件的执行器应在持久
claim之后调用，并将`TransportError.receipt.http_diagnostic`随原回执保存。
本版的替身集成测试验证了这个保存路径，也验证首项失败后其余203项未发。

HTTP错误中的`usage`或`cost=0`不作为可靠账单，故不会自动清除未知费。
本次修复不能恢复旧执行器已丢弃的响应。错误诊断字段没有被装入旧冻结入口。

## 只读检查入口

```powershell
.\.venv\Scripts\python.exe scripts/check_e1_t72.py
.\.venv\Scripts\python.exe scripts/check_e1_t72.py --online-preflight
```

默认不联网、不读取`.env`或进程Key，也不创建campaign。它复验旧源码/请求封印，
核验已存在账本及请求/回执摘要；账本使用只读、immutable连接，若存在活动SQLite日志则拒绝，
不会忽略未合入的数据。`--online-preflight`只允许无凭据公开模型目录与所选端点GET。

`ok=true`表示检查正常完成；`ready_to_send=false`表示不能发送。存在阻塞时退出码为2，
检查本身错误为1。本入口没有`--execute`、读取Key或切换run ID的选项；
不要再用T7.1的旧`ready_to_send`推导实际请求可发送。

## 复跑需要满足什么

| 条件 | 所需证据与处理 |
| --- | --- |
| 原失败可追溯 | 保留原1失败/203未发、204分母及封印；不打开旧账本的停止开关 |
| 原费用核清 | 用可归属原请求的供应商记录建立独立对账补充，绑定原请求/回执SHA；保留旧NULL与未知状态原件 |
| 实际参数有依据 | 当前原参数缺parallel支持证据。须取得实际字段支持证据，或另版登记准确请求差异；不能只在预检时忽略 |
| 新执行与历史关联 | 新协议必须引用原失败与费用，定义此次是全量新版本还是其他前瞻处理；不能换ID重置预算或混合最佳结果 |
| 费用仍受原上限约束 | 已知历史费用加拟执行整阶段预留仍须满足E1 0.5美元、裁判0.5美元、累计2美元停、总5美元；未知费未清不能以规划预留代替实费 |
| 发送前再核验 | 修复源码/请求指纹、当前端点/参数/价格、完整库存预算全部通过后，才读取已授权Key、持久claim、发送一次 |
| 裁判独立准入 | 全部实际生成原件通过后，再按所有实际文本核算完整裁判预留；不足0.5美元即停，不删批次、不换人工40条抽样 |

原Key读取许可和已有费用批准持续有效，无须重新询问同一权限。它们不取消冻结的
未知结果停发规则。当前空日志、账户总用量、当前Key有效或HTTP类别均不足以证明旧单笔费用；
本修复不提供`accept_unknown`、重试或重置开关。

如果后续选择省略`parallel_tool_calls`，必须在新登记中明确所有请求SHA变化，并保留强制
`aria_turn_decision`及单结果校验。这是后续候选方案，本版实际字段未改，不能直接用于原实验。
不为消除当前阻塞而启用备用供应商或放宽数据隐私设置。

## 验证与范围

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_t72_policy.py tests/test_t72_transport.py tests/test_e1_t72.py -q
```

软件测试覆盖实际字段缺失、修改后指纹失效、状态/标识脱敏、超长/超时/截断/转义回显、
原传输兼容、失败落盘/未知费停发、只读资格检查及修复封印。所有HTTP均为本地替身，
不证明在线路由可用或人物效果。当前真实核查输出、具体测试数量及遗留条件只在PROGRESS维护。

新修复没有扩大模型/TTS/通知调用、E2或公开发布范围。更新后的发布树需要重新审查，
不能继承任务06候选包的最终安全结论。
