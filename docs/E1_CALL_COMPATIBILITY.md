# E1实际调用兼容修订（T7.4）

当前进度只见[PROGRESS](PROGRESS.md)。本修订解决实际请求与当前OpenRouter端点工具选择能力冲突；不改旧冻结、失败原件、费用或判分参考。

## 已定位的问题与修复

1. 原预检先移除parallel字段再查能力，却发送原字段，检查内容与实际请求不一致。T7.3已省略该字段。
2. 原预检只看tool_choice字段存在，没有查取值。当前Alibaba端点明确auto=true、function=false、required=false，旧强制aria_turn_decision与之冲突。
3. T7.4实际请求使用auto并省略parallel。题目、提示、工具schema、模型、Alibaba、禁止fallback、require_parameters及data_collection=deny均保持。204个位置、68个题目/臂组合、65个不同请求重新计算SHA并关联原T6 SHA，不冒充旧wire相同。
4. 发送入口在读取Key前用同一份两次公开GET响应检查全部65个实际wire：字段＋选择模式、端点严格整数status、模型/供应商/版本、价格档、输入输出限制及reasoning.mandatory=false。检查结果绑定实际SHA并交给发送适配，不再只新增一个旁路诊断。
5. HTTP失败在同次响应中保留固定原因分类、状态和合法标识，区分参数、数据政策/隐私、模型、鉴权、余额和限流。错误原文、密钥、隐藏推理不落盘；未知/歧义明确记录，不以分类证明旧404唯一原因或零费用。

代码入口：[实际wire与全库存检查](../scripts/e1_compat.py)、[受控执行](../scripts/run_e1_compat.py)、[安全错误分类](../scripts/t74_transport.py)。公开登记在experiments/t74_20261001。

## 工具与证据合同

auto意味着服务端可以选择不用工具。因此返回后仍要求恰好一个aria_turn_decision原生工具结果、空content、合法可解析arguments、匹配模型/供应商、完整usage/cost和原token上界。无工具、自然文本、多工具、格式/保存失败均保存失败与费用、保留未发位置、不自动重试，不将自然文本转换为假工具成功。

参数兼容不保证账户/Key可路由。公开端点没有数据政策证明字段；deny由服务端强制，不因为404而删掉。reasoning.exclude只控制隐藏，不能单独证明关闭；请求enabled=false且mandatory=false预检，响应中仍核有无隐藏推理/reasoning tokens。

这是前瞻调用修订，不是题目或效果改写。之后的真实模型结果属于新版。软件替身只证明发送/保护链路；HTTP成功亦不能证明人格理解改善。

## 使用与费用

使用已冻结本机环境：

```powershell
.\.venv\Scripts\python.exe scripts/run_e1_compat.py
.\.venv\Scripts\python.exe scripts/run_e1_compat.py --online-preflight
```

默认离线；公开预检仅两次无凭据GET，不读.env、不建campaign。只有源码、历史、费用和实际参数门都过后，--execute --allow-local-env才允许仅提取本机.env的OPENROUTER_API_KEY并先持久claim再单次POST。现有目录拒绝再次启动。

当前两次失败各有一笔NULL费用；用户明确接受的原1笔例外保持精确范围，新1笔未自动扩用。两个旧账本全部原字节保持停止。参数门通过也不会跳过新未知费用门，入口没有ignore/retry开关。E1阶段0.5、裁判0.5、已知累计2停/总5美元不变；没有另开探针、TTS、E2或真实通知。裁判须完整生成原件及全量实际文本预算通过后另行执行。

## 官方依据

- [当前型号端点能力](https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints)：字段支持与auto/function/required模式分别提供，取发送前时点证据。
- [工具选择说明](https://openrouter.ai/docs/guides/features/tool-calling)：auto让模型选择工具，不能当强制工具保证。
- [参数及数据政策路由](https://openrouter.ai/docs/guides/routing/provider-selection)：require_parameters和deny均影响可选端点，须继续保留策略。
- [ZDR与不训练的区别](https://openrouter.ai/docs/guides/features/zdr)：缺ZDR旁证不能推断deny不兼容，也不能声称数据完全不保留。

本轮软件和真实执行情况见PROGRESS，不在本页维护第二份最新总表。
