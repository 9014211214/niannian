"""Installable, local-only companion Web entry point."""
from __future__ import annotations

import argparse
import json
from getpass import getpass
from pathlib import Path
from typing import Optional

from ._env import load_local_env


def web_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aria-web", description="启动 Aria 老人陪伴 Web；默认本地占位")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--minimal", action="store_true", help="兼容旧命令；当前唯一入口已是最小陪伴 Demo")
    parser.add_argument("--enable-model-calls", action="store_true", help="显式启用有费用的在线及会后模型；自动整理还须在本人页面开启；默认不读取.env")
    parser.add_argument("--minimal-data-dir", type=Path, default=Path("data/minimal_demo"), help="独立虚构资料目录；本地示意与真实生成分开保存")
    parser.add_argument("--model-max-requests", type=int, help="真实模式必填：全部职责累计调用尝试上限，重启不归零")
    parser.add_argument("--model-stop-cost-usd", help="真实模式必填：已观测USD费用停止阈值；不是账单硬上限")
    parser.add_argument("--model-reservation-policy", type=Path,
                        help="真实模式必填：已核对并冻结的价格/端点预留策略JSON")
    parser.add_argument("--care-setup-admin", metavar="USERNAME", help="交互创建护理员账号并分配虚构陈师傅，无默认密码")
    parser.add_argument("--release-profile", choices=("release_deepseek_v1", "release_v1", "release_max_v1", "legacy"), default=None,
                        help="真实模式默认DeepSeek V4.1 Flash 1017＋Sonnet；其他配置用于历史复验；预览始终不调用模型")
    parser.add_argument("--online-models-metadata", type=Path, help="首发真实模式：已核对的公开models元数据JSON")
    parser.add_argument("--online-endpoints-metadata", type=Path, help="真实模式：所选在线模型的公开端点元数据JSON")
    parser.add_argument("--offline-endpoints-metadata", type=Path, help="所选会后模型的公开端点元数据")
    from .engine.policy.memory_structured import OFFLINE_ROUTES, MODEL, PROVIDER
    parser.add_argument("--offline-model", choices=sorted({model for model, _ in OFFLINE_ROUTES}), default=MODEL,
                        help="显式选择会后模型；默认保留历史Sonnet路由")
    parser.add_argument("--offline-provider", choices=sorted({provider for _, provider in OFFLINE_ROUTES}), default=PROVIDER,
                        help="须与会后模型、公开端点和费用策略一致")
    parser.add_argument("--safety-delivery-version", choices=("legacy", "conditional_context_v1", "task_context_v2", "source_response_v3", "source_response_v4", "minimal_response_v1"),
                        default=None, help="仅历史复验或明确配置时覆盖交付版本")
    parser.add_argument("--safety-response-instruction-version", choices=("legacy", "conditional_response_v1", "conditional_response_v2", "conditional_response_v3", "conditional_response_v4", "compact_response_v1"),
                        default=None, help="仅历史复验或明确配置时覆盖提示版本")
    parser.add_argument("--premodel-safety-version", choices=("legacy", "task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4"),
                        default=None, help="仅历史复验或明确配置时覆盖前置保护版本")
    from .engine.policy.online_decision import ONLINE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE
    parser.add_argument("--online-decision-version", choices=sorted(ONLINE_DECISION_VERSIONS),
                        default=None, help="默认使用发布配置的完整协议；legacy默认v3")
    args = parser.parse_args(argv)
    from .release import engine_options, ONLINE_VERSION, ONLINE_MODEL, ONLINE_PROVIDER, DEFAULT_RELEASE_PROFILE
    from .engine.policy.online_decision import ONLINE_DECISION_VERSION_V7 as MAX_ONLINE_VERSION
    from .engine.policy.online_decision import EVIDENCE_DECISION_VERSIONS
    profile_explicit = args.release_profile is not None
    args.release_profile = args.release_profile or DEFAULT_RELEASE_PROFILE
    if ((args.offline_model, args.offline_provider) not in OFFLINE_ROUTES
            or (args.offline_model, args.offline_provider) != (MODEL, PROVIDER)
            and args.release_profile != "release_deepseek_v1"):
        parser.error("会后路由须成对匹配；DeepSeek会后仅适用于release_deepseek_v1")
    selected_online_version=MAX_ONLINE_VERSION if args.release_profile in ("release_max_v1", "release_deepseek_v1") else ONLINE_VERSION
    explicit_versions = any(getattr(args, key) is not None for key in (
        "safety_delivery_version", "safety_response_instruction_version", "premodel_safety_version", "online_decision_version"))
    # An explicit historical protocol is a compatibility request, not a silent
    # promotion of that candidate to the normal first-release entry point.
    if explicit_versions and args.release_profile in ("release_v1","release_max_v1", "release_deepseek_v1"):
        selected = engine_options()
        selected["online_decision_version"] = selected_online_version
        if any(getattr(args, key) not in (None, value) for key, value in selected.items()):
            if profile_explicit:
                parser.error("所选发布配置须使用完整版本组合；历史复验请显式选择legacy")
            args.release_profile = "legacy"
    defaults = engine_options() if args.release_profile in ("release_v1","release_max_v1", "release_deepseek_v1") and args.enable_model_calls else {}
    for key in ("safety_delivery_version", "safety_response_instruction_version", "premodel_safety_version"):
        if getattr(args, key) is None:
            setattr(args, key, defaults.get(key, "legacy"))
    args.online_decision_version = args.online_decision_version or (
        selected_online_version if defaults else "online_turn_decision_v3")
    if args.release_profile == "release_v1" and args.enable_model_calls:
        if args.online_models_metadata is None or args.online_endpoints_metadata is None:
            parser.error("首发真实模式须提供已核对的--online-models-metadata和--online-endpoints-metadata；不会自动调用模型或切换端点")
        try:
            from .engine.policy.online_structured import build_online_structured_route
            args.online_structured_route = build_online_structured_route(
                args.online_models_metadata.read_bytes(), args.online_endpoints_metadata.read_bytes(),
                model=ONLINE_MODEL, provider=ONLINE_PROVIDER)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if args.release_profile in ("release_max_v1", "release_deepseek_v1") and args.enable_model_calls:
        if any(value is None for value in (args.online_models_metadata,args.online_endpoints_metadata,args.offline_endpoints_metadata)):
            parser.error("真实配置必须提供模型、所选在线和会后端点的公开元数据")
        try:
            from .release import build_release_online_route
            from .engine.policy.memory_structured import build_memory_structured_route
            args.online_structured_route=build_release_online_route(
                args.online_models_metadata.read_bytes(),args.online_endpoints_metadata.read_bytes(),
                release_profile=args.release_profile)
            args.memory_structured_route=build_memory_structured_route(
                args.online_models_metadata.read_bytes(),args.offline_endpoints_metadata.read_bytes(),
                model=args.offline_model, provider=args.offline_provider)
        except (OSError,ValueError) as exc:parser.error(str(exc))
    if args.premodel_safety_version != "legacy":
        from .engine.risk.premodel_task import validate_premodel_configuration
        try:
            if not args.enable_model_calls:
                raise ValueError("premodel_task_requires_model_calls")
            validate_premodel_configuration(args.premodel_safety_version, args.safety_delivery_version,
                args.safety_response_instruction_version, args.online_decision_version)
        except ValueError as exc:
            parser.error(str(exc))
    if args.safety_delivery_version != "legacy" and (
            not args.enable_model_calls or args.online_decision_version not in EVIDENCE_DECISION_VERSIONS):
        parser.error("安全交付候选须显式启用模型并选择在线v4；不会自行改变模型协议")
    if args.safety_response_instruction_version != "legacy" and (
            not args.enable_model_calls or args.online_decision_version not in EVIDENCE_DECISION_VERSIONS):
        parser.error("安全回应指令候选须显式启用模型并选择在线v4；不会自行改变模型协议")
    if args.enable_model_calls:
        from .engine.policy.minimal_calls import validate_limits
        try:
            validate_limits(args.model_max_requests, args.model_stop_cost_usd)
            from .engine.policy.reserved_calls import validate_reservation_policy
            if args.model_reservation_policy is None:
                raise ValueError("真实模型必须提供发送前费用预留策略。")
            args.reservation_policy = validate_reservation_policy(
                json.loads(args.model_reservation_policy.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif (args.model_max_requests is not None or args.model_stop_cost_usd is not None
          or args.model_reservation_policy is not None):
        parser.error("模型限额必须与--enable-model-calls同时提供，不会自行启用模型")
    return _minimal_main(args)


def _minimal_main(args) -> int:
    from .runtime import create_minimal_clients
    from .voice.minimal import MinimalWebApp, make_minimal_server

    # Merely finding a configured key never enables models or optional services.
    if args.enable_model_calls:
        load_local_env()
    if args.release_profile == "release_v1":
        from .release import create_release_clients
        clients = create_release_clients(
            model_calls_enabled=args.enable_model_calls,
            online_structured_route=getattr(args, "online_structured_route", None))
    elif args.release_profile in ("release_max_v1", "release_deepseek_v1"):
        from .release import create_max_release_clients, create_deepseek_release_clients
        create_clients = create_deepseek_release_clients if args.release_profile == "release_deepseek_v1" else create_max_release_clients
        clients=create_clients(model_calls_enabled=args.enable_model_calls,
            online_structured_route=getattr(args,"online_structured_route",None),
            memory_structured_route=getattr(args,"memory_structured_route",None))
    else:
        clients = create_minimal_clients(
            model_calls_enabled=args.enable_model_calls,
            online_decision_version=args.online_decision_version)
    app = MinimalWebApp(args.minimal_data_dir, clients=clients,
                        model_max_requests=args.model_max_requests,
                        model_stop_cost_usd=args.model_stop_cost_usd,
                        model_reservation_policy=getattr(args, "reservation_policy", None),
                        **({"safety_delivery_version": args.safety_delivery_version}
                           if getattr(args, "safety_delivery_version", "legacy") != "legacy" else {}),
                        **({"safety_response_instruction_version": args.safety_response_instruction_version}
                           if getattr(args, "safety_response_instruction_version", "legacy") != "legacy" else {}),
                        **({"premodel_safety_version": args.premodel_safety_version}
                           if getattr(args, "premodel_safety_version", "legacy") != "legacy" else {}))
    if args.care_setup_admin:
        try:
            password = getpass("设置护理员密码（至少12字符）：")
            repeated = getpass("再次输入密码：")
            if password != repeated:
                raise ValueError("两次密码不一致，未创建账号。")
            app.care.bootstrap(args.care_setup_admin, password, ["fictional-chen"])
        except (ValueError, PermissionError, EOFError) as exc:
            app.close()
            print(f"护理员初始化未完成：{exc}")
            return 1
        finally:
            password = repeated = None
    server = make_minimal_server(app, args.host, args.port)
    print(app.state()["notice"])
    print(f"最小体验入口：http://{args.host}:{server.server_port}")
    print("仅虚构角色资料；封存后可手动整理，或由本人在页面明确开启自动整理。停止服务不会启动整理。护理通知仅本地模拟。")
    print(f"护理员后台：http://{args.host}:{server.server_port}/care（需先用--care-setup-admin创建账号）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        except Exception:
            print("停止时封存失败；请检查本地存储，不能保证未保存的内容可恢复。不会启动会后模型。")
            return 1
        print("已停止；本地封存记录保留，下次可读取。")
    return 0
