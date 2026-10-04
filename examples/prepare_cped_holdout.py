"""Prepare CPED holdout candidates locally; never calls a model or downloads data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from aria.bench.cped_holdout import (
    DEFAULT_SEED, assert_aggregate_only_preparation, prepare_cped_holdout,
)
from aria.bench.provenance import file_fingerprint
from aria.bench.cped_source import load_cped_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "CPED 本地留出候选构造与表面捷径检查；仅汇总输出，无模型调用、不下载。"
    ))
    parser.add_argument("--data", required=True, help="本地 CPED CSV 或 split 文件目录")
    parser.add_argument("--identity-terms", type=Path, help="本地补充名称 JSON 字符串数组；仍需人工审阅完整性")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, help="新建汇总 JSON，已有文件拒绝覆盖")
    parser.add_argument("--acknowledge-unverified-data-license", action="store_true",
                        help="运行者已复核当前上游条款；不授予权利")
    parser.add_argument("--accept-aggregate-only", action="store_true",
                        help="接受不公开台词、人物、逐例结果及可回链标识")
    args = parser.parse_args(argv)
    if not args.acknowledge_unverified_data_license or not args.accept_aggregate_only:
        parser.error("须先核对数据条款并显式接受 aggregate-only 边界；未读取数据。")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed 必须为 32 位非负整数；未读取数据。")
    if args.output is not None and args.output.exists():
        parser.error("输出文件已存在；不会覆盖旧审计或数据。")
    try:
        terms = json.loads(args.identity_terms.read_text(encoding="utf-8-sig")) if args.identity_terms else ()
        if not isinstance(terms, (list, tuple)) or any(not isinstance(term, str) or not term.strip() for term in terms):
            raise ValueError("invalid identity inventory")
        corpus = load_cped_corpus(args.data, acknowledge_unverified_data_license=True)
        prepared = prepare_cped_holdout(
            corpus.rows, terms_review_acknowledged=True, aggregate_only_accepted=True,
            extra_identity_terms=terms, seed=args.seed,
        )
    except (OSError, ValueError, KeyError, PermissionError, UnicodeError):
        # Source errors may contain restricted text/paths. Do not echo them.
        parser.error("本地数据或名称清单读取/结构检查失败；未导出原文，也未调用模型。")
    assert_aggregate_only_preparation(prepared.report)
    code_root = Path(__file__).resolve().parents[1]
    report = {
        "preparation": prepared.report,
        "source_file_sha256": sorted(corpus.sha256.values()),
        "implementation_sha256": {
            name: file_fingerprint(code_root / name) for name in (
                "aria/bench/cped_holdout.py", "aria/bench/cped_source.py",
                "examples/prepare_cped_holdout.py",
            )
        },
        "data_rights_status": "NOASSERTION_unverified_operator_acknowledged",
        "private_working_cases_exported": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0 if prepared.report["status"] == "prepared_pending_identity_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
