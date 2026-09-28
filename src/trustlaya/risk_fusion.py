"""Transparent score fusion alternatives; outputs are uncalibrated policy scores."""

from .evidence import PII_TYPES, SECRET_TYPES


def fuse(scores, evidence=(), metadata=None, agent=None, method="rule_constrained"):
    metadata = metadata or {}
    agent = agent or {}
    values = [float(scores[key]) for key in ("pii", "secret", "prompt_injection",
                                                "dangerous_instruction", "privacy_risk",
                                                "security_risk", "ethics_risk",
                                                "oversight_risk", "data_governance_risk")]
    weighted = min(1.0, sum(w * v for w, v in zip(
        (0.10, 0.13, 0.18, 0.14, 0.08, 0.14, 0.06, 0.08, 0.09), values)))
    has_pii = any(item["type"] in PII_TYPES for item in evidence)
    has_secret = any(item["type"] in SECRET_TYPES for item in evidence)
    transfer = metadata.get("external_api") or any(item["type"] == "EXTERNAL_TRANSFER" for item in evidence)
    interaction = max(weighted, 0.8 if has_pii and transfer else 0.0,
                      0.95 if has_secret and transfer else 0.0,
                      0.9 if scores["prompt_injection"] >= 0.5 and metadata.get("shell") else 0.0,
                      float(agent.get("overall_agent_risk", 0.0)))
    alternatives = {
        "weighted": weighted,
        "interaction": interaction,
        "rule_constrained": max(interaction, max(values)),
    }
    if method not in alternatives:
        raise ValueError(f"Unknown fusion method: {method}")
    return {"method": method, "score": round(alternatives[method], 4),
            "alternatives": {key: round(value, 4) for key, value in alternatives.items()},
            "source": "deterministic_policy_not_probability"}
