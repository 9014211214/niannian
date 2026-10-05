# 本地打包与发布检查

当前版本、结果、阻塞和执行队列只维护在 [PROGRESS](PROGRESS.md)。本页保留可重复执行的
检查方法；源码检查、安装成功、模拟流程、真实效果及允许发布分别判断。

## 软件与故事

在本次候选源码上安装 `.[dev,secure]`，运行：

```sh
python -m ruff check aria examples scripts tests
python -m pytest -q -ra
python -m pip check
```

测试只使用合成输入与替身服务，不调用云模型。报告记录源码版本、通过/失败/跳过分母。
旧 E1 执行器绑定旧源码摘要，应拒绝修改后的生产代码；精确列出的历史绑定测试在当前
版本明确跳过，在保留的冻结源码目录另行复验。跳过不计通过，旧效果不迁移到新实现。

浏览器至少三遍完整故事：连续会话、显式会后操作、相处说明提议/确认/撤回、当前说法
优先、本轮实际输入、护理状态、退出/重启、回放无副作用。保留每遍的问题、状态和截图，
不挑最好一遍。默认占位或本地规则提议只验证软件；真实模型故事、录屏、美术/TTS、
失败后的重录须单独获得费用授权。浏览器测试不要求招募老人或采集真人资料。

## 源码 ZIP

```sh
python scripts/build_minimal_release.py --check
python scripts/build_minimal_release.py --output dist/candidate-source.zip
python scripts/build_minimal_release.py --verify dist/candidate-source.zip
```

已有输出不能覆盖，另选新文件名。构建只读取明确白名单，检查路径、大小、内容和
SHA-256，扫描后源码变化会拒绝打包。`RELEASE_MANIFEST.json` 是完整性记录，不是签名、
安全证明或效果证据；范围以脚本及包内 `RELEASE_SCOPE.md` 为准。

不包含真实 `.env`、运行资料、Git 历史、私人实验输出、下载语料或媒体；`.env.example`
只是示例配置。旧占位截图/视频留在本机，新的公开画面须独立审查。

## wheel / sdist 与安装

构建工具安装可能联网；以下步骤不上传包。使用新的输出目录，避免混入旧候选包：

```sh
python -m pip install build twine
python -m build --outdir dist/candidate-packages
python scripts/check_release_install.py --dist-dir dist/candidate-packages
```

安装检查脚本在 Windows/Linux 使用对应虚拟环境路径，要求恰好一个 wheel 和一个 sdist，
逐个检查资源、许可文件、私有目录排除及包摘要，并执行严格元数据检查。两个新环境分别
安装 wheel/sdist，核对依赖、CLI、隔离导入来源、实际本机 HTTP 页面、三次预览会话、
护理登录/老人范围/撤回，以及读取旧记录无新动作。原型模型、真实通知和 `.env` 读取
始终关闭。记录保存在新运行目录的 `result.json`；失败保留目录，不覆盖旧证据。

离线本机复用已安装依赖时，明确使用：

```sh
python -m build --no-isolation --outdir dist/local-packages
python scripts/check_release_install.py --dist-dir dist/local-packages --work-dir artifacts/local/install-check-new --reuse-local-dependencies
```

这是安装产物检查，报告标记 `local_reused_not_cold`，不能称为陌生机器联网冷安装。
`.github/workflows/test.yml` 配置 Windows/Linux × Python 3.10/3.12，全量软件测试及两个
干净安装环境；配置已写入不代表远端四组已通过。只有实际 CI 回执能证明远端运行结果。
首发最低版本不因本机使用较新 Python 而自动验收。

## 安全、许可与公开内容

- 在最终 ZIP 上执行[安全隐私审查](RELEASE_SECURITY_AUDIT.md)，逐项解释绑定当前字节的
  命中；后续修改必须重新扫描，不能继承旧包通过结论。
- 密钥扫描只覆盖选定发布树，未覆盖 Git 历史、所有个人信息或画面。公开仓库使用干净
  历史，不直接推送现有工作区。截图/视频人工核对虚构输入、无私人内容和凭据，剪辑不能
  冒充实时延迟。
- 保留 [MIT](../LICENSE)、[第三方说明](../THIRD_PARTY_NOTICES.md)及文件级声明。
  包内不带 CPED 原文、依赖二进制或私人归档；MIT 不授予外部数据权利。实际新环境的
  依赖版本与许可需核对，现有 Windows 依赖清单不是锁文件或完整跨平台 SBOM。
- 首发护理仅本地模拟通知。真实通道、测试收件人、分享范围、费用和送达联调属于另外
  授权的工作；模拟授权不能用于真实外发。
- 医疗知识表述已收窄为查看具体药物说明或咨询医生/药师；软件与许可检查不等于专业
  医学全面审核，也不能宣称可投入机构使用。

## 效果与发布决定

E1 是作者编写的虚构场景，204 个生成位置已经完成。全量裁判为未经人工校准的 Codex，
Claude API 全量裁判未执行；后来40条由本会话Claude作部分盲标注，不能称人工校准。
固定门槛的当前结果见 [PROGRESS](PROGRESS.md)及原实验报告，不称真人效果。
核心改善、v2 回复与延迟等门未通过，默认继续 `v1`＋`mode_v1`；
软件结构检查通过不能代替这些负结果。后续相处说明与调用修订属于新的生产源码，不能
继承 E1 的效果结论。

40 条人工盲评校准是可选补充，当前未进行；不作为开发或技术开源准备的阻塞，也不要求
招募老人。缺少校准时保留其限制。CPED 仅完成本机准备，可行性门已停止，没有真实 E2，
没有把 CPED 台词发云端或公开原文的授权。

E1b-R 已完成生成与首次标注，但14个可见候选全被拒收，12题均无确认说明，未测到
预定的说明使用效果；原三门不可评，详见[固定结果](E1BR_RESULTS_2026_10_05.md)。
后续修复须另版验证。原 E1 负结果保留，C-test 只验证合成会后组件，不能
代替回复效果。技术开源须明确原型及未验收边界；能否以此定位公开，仍由用户决定，不擅自改变
发布定位。目标仓库名、建仓/推送、公开部署和真实模型
演示费用按进度中的授权边界处理。本地开发、检查和打包不等于已经获准发布。
