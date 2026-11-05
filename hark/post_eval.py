"""post_eval.py WORKDIR CONFIG [random] -> validation-only: best-bleu beam sweep, top-5 checkpoint average, length-penalty sweep (beam 4). Never touches test/gold.
Writes RUN_DIR/post/summary.json."""
import json, os, subprocess, sys, glob
from pathlib import Path
wd, cfg = Path(sys.argv[1]), sys.argv[2]; rnd = len(sys.argv) > 3 and sys.argv[3] == "random"
os.chdir(wd); env = dict(os.environ, ISL_ALLOW_SOURCE_OVERLAP="1" if rnd else "0", PYTHONUNBUFFERED="1")
for line in open(".env"):
    if "=" in line: k, v = line.strip().split("=", 1); env[k] = v
import yaml
run_dir = Path(yaml.safe_load(open(cfg))["run_dir"]); post = run_dir / "post"; post.mkdir(exist_ok=True)
P = "/mnt/scratch/islenv/bin/python"; summary = {}
def ev(name, ckpt, conf):
    out = post / name
    r = subprocess.run([P, "-m", "isl_slt", "evaluate", "--config", conf, "--checkpoint", str(ckpt), "--partition", "validation", "--output", str(out)],
                       env=env, capture_output=True, text=True)
    if r.returncode: summary[name] = {"error": r.stderr[-600:]}; return
    m = json.load(open(out / "metrics.json")); s = m["metrics"].get("scores", m["metrics"])
    d = json.load(open(out / "decoding.json"))
    summary[name] = {"beams": m["selected_beams"], "scores": {k: s.get(k) for k in ("bleu4", "chrf++", "rouge_l", "wer", "ter", "meteor") if k in s},
                     "candidates": d.get("candidates")}
best = run_dir / "checkpoints" / "best-bleu.pt"
ev("best", best, cfg)
# top-5 retained epoch checkpoints by greedy val BLEU-4 -> average
scores = {}
for f in glob.glob(str(run_dir / "validation" / "epoch-*.json")):
    j = json.load(open(f)); sc = j.get("metrics", j).get("scores", {}); scores[Path(f).stem] = sc.get("bleu4", -1)
have = sorted(Path(p).stem for p in glob.glob(str(run_dir / "checkpoints" / "epoch-*.pt")))
top = sorted(have, key=lambda e: scores.get(e, -1), reverse=True)[:5]
summary["avg_members"] = {e: scores.get(e) for e in top}
if len(top) >= 2:
    avg = run_dir / "checkpoints" / "avg-top5.pt"
    r = subprocess.run([P, "-m", "isl_slt", "average", "--checkpoints", *[str(run_dir / "checkpoints" / f"{e}.pt") for e in top], "--output", str(avg)], env=env, capture_output=True, text=True)
    if r.returncode == 0: ev("avg_top5", avg, cfg)
    else: summary["avg_top5"] = {"error": r.stderr[-600:]}
for lp in (0.8, 1.2):
    c = Path("configs") / f"_lp{lp}_{Path(cfg).name}"
    c.write_text(f"extends: {Path(cfg).name}\ngeneration: {{train_beams: [1], beams: [4], length_penalty: {lp}, max_new_tokens: 128}}\n")
    ev(f"best_lp{lp}", best, str(c))
json.dump(summary, open(post / "summary.json", "w"), indent=1)
print(json.dumps({k: v.get("scores", v) if isinstance(v, dict) else v for k, v in summary.items()}, indent=1)[:3000])
