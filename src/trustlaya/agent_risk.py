"""Deterministic permission signals. Values are policy scores, not probabilities."""


PERMISSIONS = ("shell", "filesystem", "network", "database", "email", "external_api", "credential_access")


def analyze_agent(metadata=None):
    state = metadata or {}
    active = bool(state.get("agent"))
    if not active:
        return {"shell_risk": 0.0, "filesystem_risk": 0.0, "network_risk": 0.0,
                "database_risk": 0.0, "credential_risk": 0.0, "overall_agent_risk": 0.0,
                "triggers": [], "source": "deterministic_policy"}
    approval = bool(state.get("human_approval"))
    scores = {
        "shell_risk": 0.75 if state.get("shell") else 0.0,
        "filesystem_risk": 0.45 if state.get("filesystem") else 0.0,
        "network_risk": 0.65 if state.get("network") or state.get("external_api") or state.get("email") else 0.0,
        "database_risk": 0.7 if state.get("database") else 0.0,
        "credential_risk": 0.85 if state.get("credential_access") else 0.0,
    }
    triggers = []
    if state.get("credential_access") and (state.get("external_api") or state.get("network") or state.get("email")):
        triggers.append("credential_and_external_channel")
    if state.get("shell") and (state.get("network") or state.get("external_api")):
        triggers.append("shell_and_network")
    if state.get("filesystem") and state.get("database"):
        triggers.append("filesystem_and_database")
    if not approval and any(state.get(key) for key in PERMISSIONS):
        triggers.append("no_human_approval")
    overall = max(scores.values(), default=0.0)
    if "credential_and_external_channel" in triggers:
        overall = 1.0
    elif "shell_and_network" in triggers:
        overall = max(overall, 0.9)
    elif "filesystem_and_database" in triggers:
        overall = max(overall, 0.8)
    if approval:
        overall = max(0.0, overall - 0.15)
    return {**scores, "overall_agent_risk": round(overall, 4), "triggers": triggers,
            "source": "deterministic_policy"}
