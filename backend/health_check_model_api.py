"""Health check for an OpenAI-compatible Model API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def api_url(base_url: str, path: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}{path}"
    return f"{base_url}/v1{path}"


def request_json(url: str, api_key: str, timeout: float) -> tuple[int, object]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether an OpenAI-compatible Model API is reachable."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("MODEL_API_BASE", "http://172.18.60.121:4000"),
        help="API base URL (default: MODEL_API_BASE)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MODEL_API_KEY"),
        help="API key (default: MODEL_API_KEY)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10,
        help="Request timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("FAIL: MODEL_API_KEY is not set.", file=sys.stderr)
        return 2

    url = api_url(args.base_url, "/models")
    started = time.perf_counter()
    try:
        status, payload = request_json(url, args.api_key, args.timeout)
    except HTTPError as error:
        print(f"FAIL: API returned HTTP {error.code} for {url}")
        return 1
    except (TimeoutError, URLError, OSError, ValueError) as error:
        print(f"FAIL: Could not reach Model API ({error}).")
        return 1

    elapsed_ms = (time.perf_counter() - started) * 1000
    if status != 200:
        print(f"FAIL: API returned HTTP {status} for {url}")
        return 1

    model_count = len(payload.get("data", [])) if isinstance(payload, dict) else 0
    print(f"OK: Model API is healthy ({status}, {elapsed_ms:.0f} ms, {model_count} model(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
