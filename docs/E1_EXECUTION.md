# E1执行说明（T7）

当前进展与下一步只维护在[PROGRESS](PROGRESS.md)。本页说明执行接口及证据边界，
实验题目、204位置、盲评合同与判定门仍以[T6冻结](EXPERIMENT_FREEZE.md)为准。
T7不修改94份产品源码或T6封印，另以[执行登记](../experiments/t7_20260930/execution.json)
绑定执行脚本、对应测试及本说明；封印是字节一致性记录，不是第三方签名。

## 使用

在仓库根目录、T6登记的Python环境中执行：

```powershell
.\.venv\Scripts\python.exe scripts/verify_experiment_freeze.py
.\.venv\Scripts\python.exe scripts/run_e1_experiment.py
.\.venv\Scripts\python.exe scripts/run_e1_experiment.py --online-preflight
.\.venv\Scripts\python.exe scripts/run_e1_experiment.py --stage judge
```

默认仅离线核对，返回预留与阻塞原因，不读取`.env`。`--online-preflight`仅访问
OpenRouter公开模型目录和端点元数据，不带Authorization、不调用模型或账户API。
离线核对成功不等于允许实跑；应查看`ready_to_send`。

这些入口绑定作者本次实验。公开包不含私有种子和解盲原件；其他使用者可以运行
软件替身测试，独立实验证据需另登记种子及封印，不能冒充本次实跑或复用其费用总账。

`--execute`是显式付费入口，但必须先通过T6、T7字节复验、环境、整阶段预算、
端点及token上界门；被门阻止返回退出码2，在读取密钥前结束。
本修订尚无可验证的服务端输入token/模板上界，因此没有可以绕过该门的CLI参数。
这项缺口需要补充可复核的上界方法和另版执行登记，不能删除检查来启动。

公开端点复核[回执](../experiments/t7_20260930/preflight.json)中，所选Alibaba端点
尚未在`supported_parameters`列出`parallel_tool_calls`。这表示支持证据不足，
不表示已证明服务不支持。OpenRouter的[通用参数文档](https://openrouter.ai/docs/api_reference/parameters)
列有此参数，但通用文档不能单独证明该固定路由的行为；不悄悄删掉冻结请求字段。

## 执行与停止

`run_e1_experiment.py`协调四个职责：冻结请求重建、`t7_transport.py`单次传输、
`t7_ledger.py`持久费用总账、`t7_judge.py`盲化及固定分母汇总。
本地替身可以验证这些路径，不能证明真实模型输出、费用或陪伴效果。

E1按冻结顺序串行204位置；实际payload须匹配各位置指纹。每次先保存请求、
将claim提交到SQLite，再发送；保存可见回执并同步磁盘后记录费用。
仅接收单个原生`aria_turn_decision`工具结果，保留raw及生产护栏后的delivered。
不记录供应商隐藏推理；异常泄露字段会清洗并停发，不能当有效输出继续评分。

超时、429、缺费用、缺身份、身份漂移、截断、结构失败、预留超限、保存失败均停止。
不重试、不换ID重跑，不因失败缩小204分母。已知费用保留；未知费用为未知，不能填0。
强制退出留下的started位置也视为未决。回执保存失败后仍尽量保留已知费用，
SQLite写失败另存停止标记，需人工核对原件后另行决定，不能自动续跑。

实验私有目录固定为仓库同级的`aria-private-experiments/t6_20260930_v1/`。
其中`campaign.json`绑定T6和T7封印，`campaign.sqlite3`及其`.stopped`文件、
请求/回执/结果文件须一起保留。没有run-id或输出路径参数来绕开总账；已有目录
缺数据库或封印不符即停止。目录必须由本机用户保护，不能抵御用户主动删除或改写账本。

预算仍是E1阶段0.5美元、裁判阶段0.3美元、累计2美元停下报告、总额5美元。
每个阶段先核算完整库存；总账不是服务商硬账单封顶。E2不允许注册或发送。

## 裁判与人工空表

裁判读取完整204条已保存生成结果，逐条核对总账、请求指纹和回执SHA；结果缺失或
被改动就停。不接收用户另传的一组“较好答案”。三个独立私有随机流控制raw、
delivered、配对盲化；种子文件必须匹配T6承诺。只将盲化payload发给Claude，
不发送臂名、模型名、内部动作、种子或解盲映射。

保留raw204、delivered204及27个配对比较位置。只有待审原文与上下文/判据逐字相同
才复用raw标签，并记录来源；改变后的delivered单独评。严格解析0/1/null，拒绝
缺失/重复/多余ID、布尔值和解释文字。缺失仍为不可评，不写成正确或零分。
使用T6全部固定判定门，人工校准未完成不能切换产品默认策略。

40条人工抽样与原空表来自T6私有归档，先校验SHA、坐标及ID，再另存盲化待评表。
保留原抽样次序；不可评位置也不补抽。人工标签始终为空，不复制Claude判断。

本修订对每批采用完整payload的UTF-8字节数加8192个模板预留token，并按512输出
上限计算全量费用。**8192是T7的保守预留政策，不是T6固定值、实测token或供应商保证。**
完整raw与配对至少19批，即使内容字节计0，仅这一预留与输出上限就为0.408576美元，
高于0.3美元；实际内容或改变后的delivered还会增加预留。因此当前政策无法放行
全量裁判，但不能由此断言实际Claude账单必超0.3美元，也不据此自动增加预算。
先核实可辩护的token/模板核算依据，再根据真实待评文本核算；不删批次或缩分母凑预算。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_e1_execution.py tests/test_t7_ledger.py tests/test_t7_transport.py tests/test_t7_judge.py tests/test_experiment_freeze.py -q
```

替身验证包括204位置、停发/重启、费用未知、保存失败、身份漂移、盲化泄漏、严格标签、
固定统计门及人工空表。通过数和失败原件在PROGRESS按版本记录；本页不维持另一份
“最新成绩”。真实模型、人工校准、实际浏览器故事和发布验收仍需各自的证据。
