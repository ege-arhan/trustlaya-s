"""One-command Docker bypass harness with fake target and synthetic keys."""

import json
import os
import secrets
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "demo/compose.isolated.yaml"
PROJECT = "trustedge-isolated"


def run(*args, env, check=True):
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"container command failed: {' '.join(args[:3])}; "
                           f"exit={result.returncode}; detail={result.stderr[-500:]}")
    return result.stdout.strip()


def main():
    env = dict(os.environ)
    env.update(TRUSTLAYA_SHARED_KEY=secrets.token_urlsafe(32),
               TRUSTEDGE_ADAPTER_KEY=secrets.token_urlsafe(32),
               TRUSTEDGE_TARGET_KEY=secrets.token_urlsafe(32))
    compose = ("docker", "compose", "-f", str(COMPOSE), "-p", PROJECT)
    try:
        run(*compose, "up", "-d", env=env)
        target_id = run(*compose, "ps", "-q", "target", env=env)
        target_ip = run("docker", "inspect", "-f",
                        "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                        target_id, env=env)
        for attempt in range(20):
            first = run(*compose, "exec", "-T", "-e", f"TARGET_IP={target_ip}",
                        "agent", "python", "demo/isolated_agent.py", "initial",
                        env=env, check=False)
            if first.startswith("{"):
                result = json.loads(first)
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("isolated services failed to start")
        assert all(result.values())

        def target_state():
            return json.loads(run("docker", "exec", target_id, "python", "-c",
                                  "import json,urllib.request; print(json.dumps(json.load("
                                  "urllib.request.urlopen('http://127.0.0.1:8767/count'))))",
                                  env=env))

        # Read target from inside its private network; agent has no path there.
        before = target_state()
        assert before == {"count": 2, "texts": ["Summarize this harmless text.",
                                               "Send [REDACTED] to external API"]}
        run("docker", "stop", f"{PROJECT}-adapter-1", env=env)
        adapter_outage = json.loads(run(*compose, "exec", "-T", "agent", "python",
                                        "demo/isolated_agent.py", "adapter_outage", env=env))
        run("docker", "stop", f"{PROJECT}-gateway-1", env=env)
        gateway_outage = json.loads(run(*compose, "exec", "-T", "agent", "python",
                                        "demo/isolated_agent.py", "gateway_outage", env=env))
        after = target_state()
        assert adapter_outage["adapter_outage_denied"]
        assert gateway_outage["gateway_outage_denied"]
        assert after == before
        print(json.dumps({**result, **adapter_outage, **gateway_outage,
                          "target_count_after_outages": after["count"]}))
    finally:
        run(*compose, "down", env=env, check=False)


if __name__ == "__main__":
    main()
