"""DirectorBench results analysis. Reads downloaded Kaggle run outputs and
produces a results table + QC false-alarm/miss scatter plot.

Expected layout: results/<slug>/*.{csv,json,jsonl} with per-model files.
Filenames should contain the model slug; each record needs item_id + score.
Never invents data: missing files/models are reported, not filled.
"""
import csv
import json
import os
import re
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SLUGS = ["directorbench-promptcraft-v1", "directorbench-shotlist-v1",
         "directorbench-qcjudge-v2"]
TASK_SHORT = {"directorbench-promptcraft-v1": "PromptCraft",
              "directorbench-shotlist-v1": "ShotList",
              "directorbench-qcjudge-v2": "QCJudge"}
MODELS = ["claude-opus-5-5-default", "gpt-6.1-sol", "gpt-5.6-sol",
          "grok-4.6", "deepseek-r1-0528", "qwen3-next-80b-a3b-instruct"]


def load_records(path):
    recs = []
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                recs.append(row)
    elif path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if isinstance(data, list):
            recs = data
    return recs


def get_score(r):
    for k in ("score", "Score", "SCORE"):
        if k in r and r[k] not in (None, ""):
            try:
                return float(r[k])
            except (TypeError, ValueError):
                pass
    return None


def get_item_id(r):
    for k in ("item_id", "itemId", "id", "Item"):
        if k in r:
            return str(r[k])
    return None


def find_model_file(task_dir, model):
    for fn in sorted(os.listdir(task_dir)):
        low = fn.lower()
        # match model slug loosely: opus, sol61, sol56, grok, deepseek, qwen
        if model.split("-")[0] in low or model.replace(".", "") in low.replace(".", ""):
            if low.endswith((".csv", ".json", ".jsonl")):
                return os.path.join(task_dir, fn)
    return None


def main():
    summary = {}   # task -> model -> {n, mean}
    missing = []
    for slug in SLUGS:
        tdir = os.path.join(RESULTS, slug)
        summary[slug] = {}
        if not os.path.isdir(tdir):
            missing.append(f"{slug}: no results dir")
            continue
        for model in MODELS:
            fp = find_model_file(tdir, model)
            if not fp:
                missing.append(f"{slug}/{model}: no file")
                continue
            recs = load_records(fp)
            scores = [(get_item_id(r), get_score(r)) for r in recs]
            scores = [(i, s) for i, s in scores if s is not None]
            if not scores:
                missing.append(f"{slug}/{model}: file {os.path.basename(fp)} has no scores")
                continue
            mean = sum(s for _, s in scores) / len(scores)
            summary[slug][model] = {"n": len(scores), "mean": round(mean, 4),
                                    "file": os.path.basename(fp)}
    # markdown table
    lines = ["| Model | " + " | ".join(TASK_SHORT[s] for s in SLUGS) + " |",
             "|" + "|".join(["---"] * (len(SLUGS) + 1)) + "|"]
    for m in MODELS:
        row = [m]
        for s in SLUGS:
            d = summary[s].get(m)
            row.append(f"{d['mean']:.3f} (n={d['n']})" if d else "NO DATA")
        lines.append("| " + " | ".join(row) + " |")
    table_md = "\n".join(lines)
    print(table_md)
    print()
    if missing:
        print("MISSING / UNPARSEABLE:")
        for x in missing:
            print(" -", x)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_table.md"), "w") as f:
        f.write("# DirectorBench results (real runs)\n\n" + table_md + "\n")
    # QC scatter: false alarms vs misses needs per-item verdicts; compute if possible
    try:
        make_qc_scatter(summary)
    except Exception as e:
        print("scatter skipped:", e)


def make_qc_scatter(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slug = "directorbench-qcjudge-v2"
    tdir = os.path.join(RESULTS, slug)
    pts = []
    for model in MODELS:
        fp = find_model_file(tdir, model)
        if not fp:
            continue
        recs = load_records(fp)
        fa = miss = n_decoy = n_fail = 0
        for r in recs:
            iid = (get_item_id(r) or "").lower()
            out = ""
            for k in ("output", "Output", "verdict", "Verdict"):
                if k in r and r[k]:
                    out = str(r[k])
                    break
            up = out.upper()
            pred_fail = "FAIL" in up
            if "decoy" in iid:  # should PASS; FAIL = false alarm
                n_decoy += 1
                if pred_fail:
                    fa += 1
            elif "twin" in iid or re.search(r"c\d+_", iid):  # FAIL twin; PASS = miss
                # twins are the FAIL items; decoys handled above
                if "twin" in iid:
                    n_fail += 1
                    if not pred_fail:
                        miss += 1
        if n_decoy and n_fail:
            pts.append((model, fa / n_decoy, miss / n_fail))
    if not pts:
        print("scatter: insufficient per-item verdict data")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, x, y in pts:
        ax.scatter(x, y, s=80)
        ax.annotate(model.split("-")[0], (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("False-alarm rate (flagged clean decoys)")
    ax.set_ylabel("Miss rate (passed real defects)")
    ax.set_title("QC Judge: paranoid vs useful (real runs)")
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.text(0.02, 0.95, "useful", transform=ax.transAxes, fontsize=10)
    ax.text(0.95, 0.95, "paranoid", transform=ax.transAxes, fontsize=10,
            ha="right")
    ax.text(0.02, 0.05, "lenient", transform=ax.transAxes, fontsize=10)
    ax.text(0.95, 0.05, "useless", transform=ax.transAxes, fontsize=10,
            ha="right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "qc_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
