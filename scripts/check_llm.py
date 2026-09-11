"""
Check that your LLM settings in .env actually work.

Run this after editing .env, BEFORE starting the app:

    python scripts/check_llm.py

It reads your .env, builds the configured provider, sends one tiny test
prompt, and tells you exactly what to fix if something is wrong.
"""

import sys
from pathlib import Path

# Make `import backend...` work when this script is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.llm_service import get_llm_provider
from backend.utils.errors import LLMError


def mask(secret: str) -> str:
    """Show enough of the key to identify it, never enough to leak it."""
    if not secret:
        return "(empty)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]} ({len(secret)} chars)"


def main() -> int:
    print("=" * 62)
    print("LLM CONFIGURATION CHECK")
    print("=" * 62)
    print(f"  LLM_PROVIDER    : {settings.LLM_PROVIDER or '(not set)'}")
    print(f"  LLM_MODEL       : {settings.LLM_MODEL or '(provider default)'}")
    print(f"  LLM_API_KEY     : {mask(settings.LLM_API_KEY)}")
    print(f"  LLM_BASE_URL    : {settings.LLM_BASE_URL or '(provider default)'}")
    print(f"  LLM_TEMPERATURE : {settings.LLM_TEMPERATURE}")
    print("-" * 62)

    if settings.LLM_PROVIDER == "mock":
        print("You are in MOCK mode.")
        print("Retrieval works, but no real LLM is called and answers are a")
        print("placeholder. To use a real model, set LLM_PROVIDER and")
        print("LLM_API_KEY in .env, then run this script again.")
        return 0

    # 1. Can we build the provider? (catches missing key, bad name, missing package)
    try:
        provider = get_llm_provider(force_reload=True)
        print(f"[1/2] Provider built OK  -> {provider.name} / {provider.model}")
    except LLMError as exc:
        print(f"[1/2] FAILED to build provider:\n      {exc.message}")
        return 1

    # 2. Can we actually reach the API? (catches bad key, no network, wrong URL)
    print("[2/2] Sending a test prompt...")
    try:
        reply = provider.generate(
            "You are a test harness. Reply with exactly one word.",
            "Reply with the single word: OK",
        )
    except LLMError as exc:
        print(f"      FAILED: {exc.message}")
        print("\nCommon causes:")
        print("  * Wrong or expired API key")
        print("  * No credit / billing not enabled on the account")
        print("  * Model name not available to your account (check LLM_MODEL)")
        print("  * Wrong LLM_BASE_URL for this provider")
        print("  * No internet connection or a proxy blocking the request")
        return 1

    print(f"      Model replied: {reply!r}")
    print("-" * 62)
    print("SUCCESS - your LLM is configured correctly.")
    print("Start the app with:  uvicorn backend.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
