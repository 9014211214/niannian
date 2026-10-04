# E1 实际参数修订（T7.3）

本登记只修复请求参数及发送适配；最新结果、费用状态和执行顺序统一见[PROGRESS](PROGRESS.md)。原T6、T7、T7.1、T7.2及失败原件保留。本版本不能借新编号解除历史未知费停发。

## 实际差异

旧T7.1实际发送parallel_tool_calls=false，但预检时删除该字段；Alibaba端点的supported_parameters没有该字段，同时请求require_parameters=true。这能证明旧预检漏查，不能补证原HTTP错误的状态码或完整根因。

T7.3在实际请求中仅省略parallel_tool_calls。204个位置、顺序、case/arm/repeat、消息、工具定义、forced tool_choice、温度、输出上限、reasoning、only Alibaba、禁止fallback、data_collection=deny均保持原值。每个新SHA绑定原SHA，清单保存于experiments/t73_20261001/requests.json；不得把另一个合法请求调换到当前位置。冻结文件变更均独立封印。

预检直接检查新实际wire的max_tokens、reasoning、temperature、tool_choice、tools，不再采用删字段的预检投影。省略并行开关不代表已禁用供应商并行行为；响应仍必须只有一个原生aria_turn_decision工具结果，否则原件留存并停止，不能挑选一个工具结果算成功。usage缺失、超界和异常费用规则继续生效。

## 本地发送路径与检查

scripts/t73_execution.py复用冻结的204项执行循环，注入新准备函数和新SHA/usage/单工具检查，底层沿用T7.2有界脱敏错误传输。测试通过临时账本与假HTTP opener检查真正序列化的POST，以及HTTP错误、响应异常、保存/源码失败时的持久停止；不访问真实模型或私有campaign。

scripts/check_e1_t73.py为只读命令：不带参数检查本地证据；加--online-preflight仅发两个无凭据公开目录GET。检查每个新请求和旧失败账本，明确区分参数修复通过与允许付费执行。当前不提供execute、Key读取或忽略未知费选项；发送适配已完成，付费装配须在历史对账及费用承接登记后连接。

全部204项新规划预留仍为0.2244美元；这是规划假设，不是供应商承诺，也不是旧失败实费。参数修复不保证实际模型能返回合格结构或产生有效陪伴效果。

## 原费用为什么还未知

原尝试时间是2026-09-30北京时间22:19（UTC14:19:08–09附近），位置e1_001/e1_t01/B0/repeat1，延迟862.605ms。请求规范SHA为111189eb00e0b7418c57c3704f3cf4bd7a42c534894715d36ea71acf37930394。原HTTP错误被关闭时没有保存状态码、响应头或正文，回执envelope为空；本地已无法恢复请求ID。最大请求的公开预检SHA不是供应商请求ID，也不是首项SHA。

本地原件、备份、日志和后续预检无可归属的供应商标识。浏览器日志空列表、当前Key有效、账户余额、规划预留均不足以证明单笔零费用。原账本保留失败1/未发203/有效0、cost_usd=NULL、未知1；不把known_cost_usd=0写成实际总费为零。

恢复执行至少需要可归属上述Key/时间/模型请求的供应商计费金额或明确不计费确认。若取得记录，独立保存对账补充及证据哈希，绑定原campaign、请求SHA、回执SHA、账本与本登记的prior_campaign证据集合；原账本和失败原件不改。当前未取得这种证据，不填虚构金额。

未来新执行须承接已核清的历史费用：历史E1实费＋新204项预留不超过E1 0.5美元，并继续裁判0.5、累计2停/总5美元。原203项未发预留不算实际消费；原失败不加入新204项有效分母，也不隐去。裁判仍需完整生成结果及实际文本的全阶段预留，不能凭当前参数通过自动开跑。

## 可转交供应商的核对内容

请在用户自己的账户中核查2026-09-30 14:19:08–09 UTC附近的POST /api/v1/chat/completions：请求模型qwen/qwen3.7-flash，指定Alibaba、require_parameters=true、parallel_tool_calls=false、禁止fallback、data_collection=deny；客户端约862.605ms收到HTTP错误，未记录状态码或generation/request ID。请提供可归属该请求的状态及计费金额，或明确确认未计费。无须发送密钥、原始对话或心理资料。

以上是待转交内容；本任务没有联系供应商或发送外部消息。此处不追加新的费用授权门，已有预算与Key提取授权持续有效；缺的是历史单笔证据及原停止规则要求的承接登记。
