# 可选阿里语音转文字

[当前进度](PROGRESS.md)是执行结果、费用预留和未完成项的唯一入口。本接入只把指定音频转成文字；原对话、会后整理模型和路由保持。网页仍使用文字输入，没有开启麦克风或自动录音。

## 本地预览

安装可选依赖后，使用单声道、16 kHz、16 位 PCM WAV，或相同规格的裸 PCM：

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[asr]"
.\.venv\Scripts\python.exe -m aria.voice.asr_cli .\sample.wav
~~~

第二条命令只检查文件和显示时长、音频指纹，不读取密钥或调用服务。默认最多10秒，可显式设定不超过60秒；音频不会自动重采样或切分。开源前验证只使用公开样本或虚构内容，不采集真人评测资料。

## 显式单次转写

项目的阿里密钥使用 `ALIYUN_DASHSCOPE_API_KEY`，通过环境变量或显式指定的本地 `.env` 提供。密钥不应写入命令参数、浏览器或日志。先把预留纳入自己的总预算，再运行：

~~~powershell
.\.venv\Scripts\python.exe -m aria.voice.asr_cli .\sample.wav --enable-asr-calls --env-file .env --max-audio-seconds 10 --fee-reserve-cny 0.10 --output-dir .\artifacts\local\asr-my-first-run
~~~

输出目录必须全新；已存在的任务目录不可覆盖或续跑。一次调用仅创建一个阿里任务，零自动重试、重连或回退，总超时90秒。固定使用北京官方接口和 `qwen-audio-3.1-asr-flash-streaming`。模型名中的 `streaming` 是实时接口需要的完整名称。

查看输出目录的 `result.json` 中 `transcript`。只有同一任务成功结束、收到最终句并且没有未完成语句，才提供可提交的文字。中间识别会单独保留；缺少最终句、任务失败、超时或任务编号冲突均不能用最后一个中间结果冒充成功。

本命令只保存转写预览，不会送进对话或启动会后整理。当前网页可以由本人核对转写后，明确提交到原有文字输入框。程序接入可使用 `submit_final_transcript`，但必须传入本进程实际完成且未修改的识别结果、当前会话编号，以及明确的 `confirmed=True`；原文字入口继续核验会话、权限及重复提交。语音识别本身不会确认偏好、批准转达或创建云模型客户端。

## 费用与限制

公开标价和免费额度条件见[阿里计价文档](https://help.aliyun.com/zh/model-studio/model-pricing)。免费余额、有效期和适用模型须在自己的账户中核实；转写成功不能证明免费。这里的 `--fee-reserve-cny` 是调用者申报的未知费用预留，通用命令不会自动核对项目总账，也不是供应商强制的费用上限。

任务结束事件未带用量时，`final_usage` 为 `null`；其他事件的最后一次累计用量另记在 `last_observed_usage`，不能把多次累计值相加。用量乘标价只能作为估算，实际收费仍以可归属账单为准。

本项目本次冒烟另由唯一执行器控制公开样本、等待、总账和原始事件保留。一次公开短音频通过，只证明本次接入可用，不证明老人语音、方言准确率或陪伴效果，也不替代三遍正式 Demo 和87题安全复测。

接口依据：[实时识别](https://help.aliyun.com/zh/model-studio/real-time-speech-recognition-user-guide) · [WebSocket API](https://help.aliyun.com/zh/model-studio/qwen-audio-asr-streaming-websocket-api)。

