"""Start the credential-owning adapter as a separate process."""

import json
import os
from pathlib import Path

from trustlaya.trusted_adapter import make_adapter


def main():
    rules_path = Path(os.environ["TRUSTLAYA_TOOL_RULES"])
    rules = json.loads(rules_path.read_text())["tools"]
    if len(rules) != 1:
        raise ValueError("adapter requires exactly one configured tool")
    rule = rules[0]
    server = make_adapter(
        host=os.getenv("TRUSTEDGE_ADAPTER_HOST", "127.0.0.1"),
        port=int(os.getenv("TRUSTEDGE_ADAPTER_PORT", "8766")),
        gateway_url=os.environ["TRUSTLAYA_GATEWAY_URL"],
        gateway_key=os.environ["TRUSTLAYA_SHARED_KEY"],
        adapter_key=os.environ["TRUSTEDGE_ADAPTER_KEY"],
        target_url=os.environ["TRUSTEDGE_TARGET_URL"],
        target_key=os.environ["TRUSTEDGE_TARGET_KEY"],
        tool={key: rule[key] for key in ("name", "operation", "target")},
        permissions=rule["permissions"])
    print("Trusted adapter ready", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
