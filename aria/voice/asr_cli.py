"""One explicit WAV/PCM transcription task, separate from dialogue inference."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import wave

from .asr import (ASRConfig, ASRError, ENDPOINT, MODEL, PCM_BYTES_PER_SECOND,
                  SAMPLE_RATE, transcribe_pcm, validate_pcm)


def read_audio(path: Path, audio_format: str, config: ASRConfig) -> bytes:
    """Read a bounded file without decoding, resampling, or correcting speech."""
    pcm_limit = int(config.max_audio_seconds * PCM_BYTES_PER_SECOND)
    # WAV headers/metadata have their own finite allowance.
    file_limit = pcm_limit + (65536 if audio_format == "wav" else 0)
    with path.open("rb") as stream:
        raw = stream.read(file_limit + 1)
    if len(raw) > file_limit:
        raise ValueError("asr_audio_file_limit_exceeded")
    if audio_format == "pcm":
        pcm = raw
    elif audio_format == "wav":
        try:
            with wave.open(io.BytesIO(raw), "rb") as audio:
                if (audio.getnchannels() != 1 or audio.getsampwidth() != 2
                        or audio.getframerate() != SAMPLE_RATE
                        or audio.getcomptype() != "NONE"):
                    raise ValueError("asr_wav_requires_16khz_16bit_mono_pcm")
                frames = audio.getnframes()
                if frames * 2 > pcm_limit:
                    raise ValueError("asr_audio_limit_exceeded")
                pcm = audio.readframes(frames)
                if len(pcm) != frames * 2:
                    raise ValueError("asr_wav_truncated")
        except (wave.Error, EOFError):
            raise ValueError("invalid_asr_wav") from None
    else:
        raise ValueError("invalid_asr_audio_format")
    validate_pcm(pcm, config)
    return pcm


def _project_key(env_file: Path | None) -> str:
    """Read only the requested variable; never load other service configuration."""
    key = os.environ.get("ALIYUN_DASHSCOPE_API_KEY", "")
    if key:
        return key
    if env_file is not None:
        with env_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                if "=" not in line or line.lstrip().startswith("#"):
                    continue
                name, value = line.split("=", 1)
                if name.strip() == "ALIYUN_DASHSCOPE_API_KEY":
                    return value.strip().strip('"').strip("'")
    raise ValueError("asr_api_key_missing")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aria-asr", description=(
        "单次阿里语音转写；默认只检查本地文件，不调用模型、不发送给对话。"))
    parser.add_argument("audio", type=Path)
    parser.add_argument("--format", choices=("wav", "pcm"), default="wav")
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--enable-asr-calls", action="store_true")
    parser.add_argument("--env-file", type=Path, help="仅真实调用读取其中ALIYUN_DASHSCOPE_API_KEY")
    parser.add_argument("--output-dir", type=Path, help="真实调用必填且须为全新目录，不覆盖旧任务")
    parser.add_argument("--fee-reserve-cny", help="真实调用必填：已纳入项目预算的人民币未知费用预留；不假设免费")
    args = parser.parse_args(argv)
    try:
        config = ASRConfig(enabled=args.enable_asr_calls,
                           max_audio_seconds=args.max_audio_seconds)
        pcm = read_audio(args.audio, args.format, config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc) if isinstance(exc, ValueError) else "asr_audio_file_unavailable")
    preview = {"model": MODEL, "endpoint": ENDPOINT, "sample_rate": SAMPLE_RATE,
               "channels": 1, "sample_width_bytes": 2, "audio_bytes": len(pcm),
               "audio_sha256": hashlib.sha256(pcm).hexdigest(),
               "audio_seconds": len(pcm) / PCM_BYTES_PER_SECOND,
               "max_tasks": 1, "automatic_retries": 0,
               "timeout_seconds": config.timeout_seconds,
               "dialogue_submitted": False}
    if not args.enable_asr_calls:
        print(json.dumps({**preview, "status": "local_preview_no_call"}, ensure_ascii=False))
        return 0
    if args.output_dir is None or args.fee_reserve_cny is None:
        parser.error("真实调用需要--output-dir和--fee-reserve-cny；未启动任务")
    try:
        reserve = Decimal(args.fee_reserve_cny)
        if not reserve.is_finite() or reserve <= 0:
            raise InvalidOperation
    except InvalidOperation:
        parser.error("asr_fee_reserve_must_be_positive")
    # A preexisting directory is never reused, even if the prior call failed.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        parser.error("asr_output_directory_not_new")
    receipt = {**preview, "started_at_utc": _now(), "status": "claimed",
               "unknown_fee_reserve_cny": str(reserve),
               "actual_charge_cny": None, "free_quota_verified": False}
    key = ""

    def save(name: str, value: dict):
        rendered = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        if key:
            rendered = rendered.replace(key, "[REDACTED]")
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")

    save("single-task-claim.json", receipt)
    result = None
    exit_code = 1
    try:
        key = _project_key(args.env_file)
        result = asyncio.run(transcribe_pcm(pcm, api_key=key, config=config))
        receipt["status"] = "completed"
        exit_code = 0
    except ASRError as exc:
        result = exc.result
        receipt.update(status="failed", error_code=exc.code)
    except KeyboardInterrupt:
        receipt.update(status="failed", error_code="asr_interrupted")
    except Exception:
        receipt.update(status="failed", error_code="asr_configuration_or_execution_failed")
    finally:
        if result is not None:
            saved = asdict(result)
            saved["transcript"] = result.transcript
            save("result.json", saved)
            receipt["task_id"] = result.task_id
            receipt["run_task_sent"] = result.run_task_sent
        receipt.update(finished_at_utc=_now(), exit_code=exit_code, can_resume=False)
        save("single-task-closure.json", receipt)
    # Transcription lives in the requested result file, not generic console logs.
    print(json.dumps({"status": receipt["status"], "exit_code": exit_code,
                      "dialogue_submitted": False}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
