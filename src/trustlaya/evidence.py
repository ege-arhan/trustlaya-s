import re
import ipaddress

PATTERNS = {
    "TC_KIMLIK": r"(?<!\d)\d{11}(?!\d)",
    "EMAIL": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "PHONE": r"(?<!\d)(?:\+?90\s?)?(?:0?5\d{2})[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)",
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "CARD": r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
    "IBAN": r"\bTR\d{2}(?:[ ]?\d{4}){5}[ ]?\d{2}\b",
    "ADDRESS": r"\b(?:adres|address)\s*[:=]\s*[^\n,;]{8,80}",
    "NAME": r"\b(?:ad soyad|full name|müşteri adı)\s*[:=]\s*[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+",
    "CUSTOMER_ID": r"\b(?:müşteri no|customer id)\s*[:=]\s*[A-Z0-9-]{5,20}\b",
    "API_KEY": r"\b(?:api[_ -]?key|token)\s*[:=]\s*[A-Za-z0-9_\-]{16,}\b",
    "BEARER": r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}",
    "PRIVATE_KEY": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "GITHUB_TOKEN": r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}\b",
    "AWS_KEY": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "PASSWORD": r"\b(?:password|parola|şifre)\s*[:=]\s*[^\s,;]{6,}",
    "DB_CREDENTIAL": r"\b(?:postgres|mysql|mongodb)://[^\s:@]+:[^\s@]+@[^\s]+",
    "ENV_SECRET": r"\b[A-Z][A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|KEY)\s*=\s*[^\s]{8,}",
    "PII_MENTION": r"\b(?:TC kimlik numara\w*|social security number|passport number|müşteri listesi|customer list)\b",
    "EXTERNAL_TRANSFER": r"\b(?:gönder\w*|aktar\w*|send|upload|exfiltrate|API'ye)\b",
}
SECRET_TYPES = {"API_KEY", "BEARER", "PRIVATE_KEY", "GITHUB_TOKEN", "AWS_KEY", "PASSWORD", "DB_CREDENTIAL", "ENV_SECRET"}
PII_TYPES = {"TC_KIMLIK", "EMAIL", "PHONE", "IP", "CARD", "IBAN", "ADDRESS", "NAME", "CUSTOMER_ID", "PII_MENTION"}
def _luhn(s):
    d=[int(c) for c in s if c.isdigit()]
    return len(d) in (13,14,15,16,17,18,19) and sum((n*2-9 if n*2>9 else n*2) if i%2 else n for i,n in enumerate(reversed(d)))%10==0

def extract(text):
    spans=[]
    for typ, pat in PATTERNS.items():
        for m in re.finditer(pat,text,re.I):
            value=m.group()
            if typ=="IP":
                try: ipaddress.ip_address(value)
                except ValueError: continue
            if typ=="TC_KIMLIK":
                d=[int(c) for c in value]
                if d[0]==0 or (sum(d[0:9:2])*7-sum(d[1:8:2]))%10!=d[9] or sum(d[:10])%10!=d[10]: continue
            if typ=="CARD" and not _luhn(value): continue
            spans.append({"type":typ,"text":value,"start":m.start(),"end":m.end()})
    return sorted(spans,key=lambda s:(s["start"],s["end"]))
