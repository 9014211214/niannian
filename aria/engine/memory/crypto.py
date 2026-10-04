"""档案静态加密(P5-#20)—— 一个环境变量开启,PersonaStore 透明加解密。

用法:设置 ARIA_PROFILE_KEY=<口令> 即开启;落盘自动加密(Fernet:
AES-128-CBC + HMAC-SHA256,PBKDF2-SHA256 从口令派生,盐随 store 目录
首次生成并存于 <dir>/.salt)。不设口令则明文(开发默认)。

迁移与红线:
- 有 key 读到明文档案 → 正常读,下次保存自动变密文(平滑迁移);
- 遇到密文但没设 key / key 不对 → **fail loud 抛错**,绝不静默当
  「新用户」处理——那会让引擎用空档案覆盖真人的心理画像;
- 依赖 cryptography(pip install aria-eldercare-agent[secure]);未装却设了 key → 报错。
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

_MAGIC = "ARIAENC1:"          # 密文头:识别用,格式版本化
_PBKDF2_ITERATIONS = 200_000

# 进程内缓存:PBKDF2 有意很慢(抗暴力),同一 key+salt 不重复派生。
_cipher_cache: dict[tuple[str, bytes], object] = {}


def profile_encryption_configured() -> bool:
    """是否已显式配置档案静态加密；不读取、派生或输出密钥。"""
    return bool(os.getenv("ARIA_PROFILE_KEY"))


def _load_salt(store_dir: Path) -> bytes:
    path = store_dir / ".salt"
    if path.exists():
        return path.read_bytes()
    store_dir.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(16)
    path.write_bytes(salt)
    return salt


def _get_cipher(store_dir: Path):
    """环境有 key 则返回 Fernet 实例,否则 None。cryptography 缺失时报错指路。"""
    key = os.getenv("ARIA_PROFILE_KEY")
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as e:                               # pragma: no cover
        raise RuntimeError(
            "ARIA_PROFILE_KEY 已设置但未安装 cryptography;"
            "运行 pip install aria-eldercare-agent[secure]"
            "（或 pip install cryptography）"
        ) from e
    import base64

    salt = _load_salt(store_dir)
    cached = _cipher_cache.get((key, salt))
    if cached is not None:
        return cached
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=_PBKDF2_ITERATIONS)
    fernet = Fernet(base64.urlsafe_b64encode(kdf.derive(key.encode("utf-8"))))
    _cipher_cache[(key, salt)] = fernet
    return fernet


def is_encrypted(text: str) -> bool:
    return text.startswith(_MAGIC)


def encrypt_if_keyed(text: str, store_dir: Path) -> str:
    """设了 key 就加密,否则原样返回(明文开发默认)。"""
    cipher = _get_cipher(store_dir)
    if cipher is None:
        return text
    return _MAGIC + cipher.encrypt(text.encode("utf-8")).decode("ascii")


def decrypt_if_needed(text: str, store_dir: Path, *, who: str = "") -> str:
    """密文→解密;明文→原样。密文而无 key / key 错 → fail loud。"""
    if not is_encrypted(text):
        return text
    cipher = _get_cipher(store_dir)
    if cipher is None:
        raise RuntimeError(
            f"档案 {who or '(unknown)'} 是加密的,但未设置 ARIA_PROFILE_KEY。"
            "拒绝按『新用户』处理(那会覆盖真实心理画像)——请提供口令。"
        )
    from cryptography.fernet import InvalidToken
    try:
        return cipher.decrypt(text[len(_MAGIC):].encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise RuntimeError(
            f"档案 {who or '(unknown)'} 解密失败:ARIA_PROFILE_KEY 不正确"
            "(或 .salt 文件被更换)。拒绝覆盖,请核对口令。"
        ) from e
