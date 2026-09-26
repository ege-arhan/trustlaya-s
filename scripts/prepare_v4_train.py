"""Build local research TRAIN from JOT without importing its overlapping JLL records."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

from run_external_real import jailbreak_samples, norm

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "benchmarks/external/predictions"
JOT = ROOT / "benchmarks/external/raw/v4/jailbreaks_over_time"


def key(text): return hashlib.sha256(norm(text).encode()).hexdigest()


def main():
    revision = subprocess.check_output(["git", "-C", str(JOT), "rev-parse", "HEAD"], text=True).strip()
    source = json.loads((JOT/"data/jailbreaksovertime.json").read_text())
    groups = defaultdict(list); benign=[]
    for row in source:
        text = row["prompt"].strip()
        if not text: continue
        if row["source"].startswith("jailbreak_chat") and row["label"] == 1:
            # Template prefix grouping prevents thousands of composed variants
            # from dominating the attack class as apparently independent seeds.
            group = hashlib.sha256(norm(text[:100]).encode()).hexdigest()
            groups[group].append(row)
        elif row["source"] == "wildchat" and row["label"] == 0 and len(text) >= 30:
            benign.append(row)
    chosen=[]
    for group, rows in sorted(groups.items()):
        # Four variants at most per human-authored jailbreak template.
        for row in sorted(rows, key=lambda r: r["uid"])[:4]:
            chosen.append({"id":row["uid"],"text":row["prompt"],"gold":1,
                           "source":"JOT/JailbreakChat composed", "group":group,
                           "label":"DIRECT_JAILBREAK_COMPOSED", "language":"unannotated"})
    hard=[r for r in benign if re.search(r"prompt injection|jailbreak|system prompt|security|safety|bypass",r["prompt"],re.I)]
    ordinary=sorted((r for r in benign if r not in hard),key=lambda r:r["uid"])[:2000]
    for row in hard+ordinary:
        chosen.append({"id":row["uid"],"text":row["prompt"],"gold":0,
                       "source":"JOT/WildChat", "group":row["uid"],
                       "label":"BENIGN_USER_REQUEST_UNREVIEWED", "language":"unannotated"})
    # Exact and approximate train-to-frozen-test screen. A related template in
    # both places is removed from TRAIN, leaving the existing frozen test intact.
    test=jailbreak_samples()[0]
    test_norm={norm(r["text"]) for r in test}
    exact=[r for r in chosen if norm(r["text"]) in test_norm]
    chosen=[r for r in chosen if norm(r["text"]) not in test_norm]
    vectorizer=HashingVectorizer(analyzer="char",ngram_range=(4,5),n_features=2**18,
                                 alternate_sign=False,norm="l2")
    reference=vectorizer.transform([norm(r["text"]) for r in test])
    candidates=vectorizer.transform([norm(r["text"]) for r in chosen])
    nn=NearestNeighbors(n_neighbors=1,metric="cosine",algorithm="brute",n_jobs=-1).fit(reference)
    distance,_=nn.kneighbors(candidates)
    near=[r for r,d in zip(chosen,distance[:,0]) if 1-float(d)>=.85]
    kept=[r for r,d in zip(chosen,distance[:,0]) if 1-float(d)<.85]
    seen=set(); dedup=[]
    for row in kept:
        row["sha256"]=key(row["text"])
        if row["sha256"] in seen: continue
        seen.add(row["sha256"]); dedup.append(row)
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/"v4_train_text_private.json").write_text(json.dumps(dedup,ensure_ascii=False,indent=2)+"\n")
    audit={"JOT_revision":revision,"JOT_source_total":len(source),
           "JOT_JailbreakChat_template_groups":len(groups),"JOT_JLL_records_excluded_by_source":sum(r["source"].startswith("jailbreak_llms") for r in source),
           "pre_screen":len(chosen)+len(exact),"train_test_exact_excluded":len(exact),
           "train_test_near_cosine_0.85_excluded":len(near),
           "train_after_screen":len(dedup),"train_class_counts":dict(Counter(r["gold"] for r in dedup)),
           "train_label_counts":dict(Counter(r["label"] for r in dedup)),
           "attack_template_groups_after_screen":len({r["group"] for r in dedup if r["gold"]==1}),
           "final_test":"JailbreakLLMs remains excluded from fitting; final 5,761-row clean cohort unchanged"}
    (ROOT/"reports/v4_training_data_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":main()
