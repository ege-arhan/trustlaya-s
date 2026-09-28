"""Read-only status command; never prints credentials."""

import argparse
import json
import os
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("TRUSTLAYA_GATEWAY_URL",
                                                "http://127.0.0.1:8765"))
    args = parser.parse_args()
    with urlopen(args.url.rstrip("/") + "/health", timeout=2) as response:
        status = json.load(response)
    print(json.dumps(status, indent=2))
    if status["gateway"] != "healthy" or status["authorization"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
