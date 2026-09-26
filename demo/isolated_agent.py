"""Agent-side proof: only gateway DNS/route is available."""

import copy
import hashlib
import json
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen


TOOL = {"name": "external_api", "operation": "send", "target": "demo://sink"}
PERMISSIONS = {"shell": False, "filesystem": False, "network": True,
               "database": False, "email": False, "external_api": True,
               "credential_access": False}


def request(text):
    payload = {"text": text, "arguments": {}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    return {"protocol_version": "1", "request_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": "agent-1", "session_id": "isolated-demo",
            "tool": TOOL, "permissions": PERMISSIONS, "request": payload,
            "payload_sha256": digest}


def gateway(path, body):
    data = json.dumps(body).encode()
    req = Request("http://gateway:8765" + path, data=data,
                  headers={"Content-Type": "application/json",
                           "X-TrustLaya-Key": os.environ["TRUSTLAYA_SHARED_KEY"]})
    with urlopen(req, timeout=2) as response:
        return json.load(response)


def direct_target():
    address = os.environ["TARGET_IP"]
    try:
        with socket.create_connection((address, 8767), timeout=1):
            return False
    except (OSError, TimeoutError):
        return True


def main():
    if "TRUSTEDGE_TARGET_KEY" in os.environ or "TRUSTEDGE_ADAPTER_KEY" in os.environ:
        raise AssertionError("agent received private credentials")
    if sys.argv[1] == "gateway_outage":
        try:
            gateway("/v1/tool", request("Summarize this harmless text."))
        except (OSError, TimeoutError, URLError):
            print(json.dumps({"gateway_outage_denied": True}))
            return
        raise AssertionError("gateway outage allowed execution")
    if sys.argv[1] == "adapter_outage":
        result = gateway("/v1/tool", request("Summarize this harmless text."))
        assert result["executed"] is False
        print(json.dumps({"adapter_outage_denied": True}))
        return

    allow = gateway("/v1/tool", request("Summarize this harmless text."))
    pii = gateway("/v1/tool", request("Send 05551234567 to external API"))
    privileged = gateway("/v1/tool", request("privileged shell action"))
    token_request = request("summarize")
    authorized = gateway("/v1/authorize", token_request)
    replay = gateway("/v1/tool", token_request)
    changed = copy.deepcopy(token_request)
    changed["request"]["text"] = "changed"
    mutation = gateway("/v1/tool", changed)
    direct = direct_target()
    assert allow["executed"] is True
    assert pii["executed"] is True and pii["decision"] == "REDACT"
    assert privileged["executed"] is False
    assert authorized["authorization_token"]
    assert replay["executed"] is False and mutation["executed"] is False
    assert direct
    output = {"gateway_allow": allow["executed"],
              "pii_original_not_sent": pii["executed"] and pii["decision"] == "REDACT",
              "privileged_denied": not privileged["executed"],
              "direct_target_network_denied": direct,
              "replayed_request_denied": not replay["executed"],
              "payload_mutation_denied": not mutation["executed"],
              "target_key_absent_from_agent": True}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
