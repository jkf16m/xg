"""Client construction for the TypeSafe Jev SDK.

The API key comes from ``TYPESAFE_API_KEY`` when set, otherwise from
``pass show jev`` as a subprocess — the same pattern the previous OpenRouter
client used for its key.
"""

import os
import subprocess

from typesafe_sdk import TypeSafeClient

ENTRY_NAME = "jev"


def get_api_key() -> str | None:
    """Return the TypeSafe API key, or ``None`` when no source provides one."""
    env = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if env:
        return env
    try:
        result = subprocess.run(  # noqa: S603
            ["pass", "show", ENTRY_NAME],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_client(model: str | None = None) -> TypeSafeClient:
    """Build a client from the resolved key and optional model.

    When no key is found, the SDK is left to raise its own clear error rather
    than failing here with a less specific one.
    """
    kwargs: dict[str, object] = {}
    key = get_api_key()
    if key is not None:
        kwargs["api_key"] = key
    if model is not None:
        kwargs["model"] = model
    return TypeSafeClient(**kwargs)
