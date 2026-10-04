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
    parser.add_argument("--enable-model-calls", action="store_true", help="显式启用有费用的在线及手动会后模型；默认不读取.env")
    parser.add_argument("--minimal-data-dir", type=Path, default=Path("data/minimal_demo"), help="独立虚构资料目录；本地示意与真实生成分开保存")
    parser.add_argument("--model-max-requests", type=int, help="真实模式必填：三职责累计调用尝试上限，重启不归零")
    parser.add_argument("--model-stop-cost-usd", help="真实模式必填：已观测USD费用停止阈值；不是账单硬上限")
    parser.add_argument("--model-reservation-policy", type=Path,
                        help="真实模式必填：已核对并冻结的价格/端点预留策略JSON")
    parser.add_argument("--care-setup-admin", metavar="USERNAME", help="交互创建护理员账号并分配虚构陈师傅，无默认密码")
    args = parser.parse_args(argv)
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
    clients = create_minimal_clients(model_calls_enabled=args.enable_model_calls)
    app = MinimalWebApp(args.minimal_data_dir, clients=clients,
                        model_max_requests=args.model_max_requests,
                        model_stop_cost_usd=args.model_stop_cost_usd,
                        model_reservation_policy=getattr(args, "reservation_policy", None))
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
    print("仅虚构角色资料；关闭/闲置只封存。会后模型须明确点击整理。护理通知仅本地模拟。")
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
