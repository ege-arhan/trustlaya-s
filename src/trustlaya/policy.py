from pathlib import Path
import yaml
from .evidence import PII_TYPES, SECRET_TYPES
from .uncertainty import should_abstain
DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"
def decide(scores, evidence, metadata=None, confidence=1.0, config=None, agent_risk=None, fusion=None, session=None):
    t = config or yaml.safe_load(DEFAULT.read_text())
    m = metadata or {}
    has_pii = any(e["type"] in PII_TYPES for e in evidence)
    has_secret = any(e["type"] in SECRET_TYPES for e in evidence)
    transfer = any(e["type"]=="EXTERNAL_TRANSFER" for e in evidence) or m.get("external_api",False)
    if has_secret: return "BLOCK", "secret_evidence"
    if has_pii and transfer: return "REDACT", "pii_external_transfer"
    if session and "credential_external_transfer_chain" in session.get("triggers", ()):
        return "BLOCK", "session_credential_exfiltration"
    if agent_risk and "credential_and_external_channel" in agent_risk.get("triggers", ()) and not m.get("human_approval"):
        return "BLOCK", "agent_credential_external_channel"
    if session and "filesystem_shell_external_chain" in session.get("triggers", ()):
        return "REVIEW", "session_tool_chain"
    if should_abstain(confidence,t["confidence"]): return "REVIEW", "uncertain"
    if scores["prompt_injection"]>=t["prompt_injection"]: return "BLOCK", "prompt_injection"
    if scores["dangerous_instruction"]>=t["dangerous_instruction"]: return "BLOCK", "dangerous_instruction"
    # A shifted secret-only score is too noisy to justify an automatic block.
    # Verified pattern evidence above can still block; model-only hits go to review.
    if scores["secret"]>=t["secret"]: return "REVIEW", "unverified_secret_score"
    if scores["pii"]>=t["pii"] and transfer: return "REDACT", "pii_external_transfer"
    if m.get("agent") and (m.get("shell") or m.get("credential_access") or (m.get("database") and m.get("network"))) and not m.get("human_approval"):
        return "REVIEW", "agent_permissions"
    if fusion and fusion["score"] >= 0.75: return "REVIEW", "fused_risk"
    if max(scores.values())>=0.7 or has_pii: return "REVIEW", "elevated_risk"
    return "ALLOW", "low_risk"
