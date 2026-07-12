#!/usr/bin/env python3
"""Identify experiment lines whose result dir is missing (failed/OOM runs) and
write per-file rerun_*.txt containing only those lines. Run-name reconstruction
mirrors each run_*.py exactly (f2tag = str(x).replace('.','p'))."""
import os, re, glob

ROOT = "/work/scitas-share/FellayMultiOmic/code/full-test-pipeline"
DATASETS = ["TCGA-BRCA", "TCGA-LGG", "TCGA-KIPAN"]
AXES = {"missing_hpgrid": "missing", "ratio_hpgrid": "ratio",
        "nomics_defaults": "nomics", "npatients_defaults": "npatients"}
TS = re.compile(r"_\d{8}_\d{6}_UTC$")

def ftag(v):   # float args passed through f2tag
    return str(float(v)).replace(".", "p")

def parse(line):
    t = line.split()
    d = {}
    i = 0
    while i < len(t):
        if t[i].startswith("--"):
            d[t[i][2:]] = t[i + 1]
            i += 2
        else:
            i += 1
    return d

def omics_tag(d):
    return "-".join(sorted(d["omics"].split(",")))

def run_name(model, axis, d):
    seed = d["random-state"]
    if model == "integrao":
        mode = d["mode"]
        c = f"__C{ftag(d['lr-C'])}" if mode == "unsupervised" else ""
        base = f"integrao__{mode}"
        if axis == "missing":
            return f"{base}__k{d['k-per-omic']}__nb{d['neighbor-size']}__emb{d['embedding-dims']}__frac{ftag(d['include-non-intersection-frac'])}{c}__seed{seed}"
        if axis == "ratio":
            return f"{base}__nb{d['neighbor-size']}__emb{d['embedding-dims']}__ratio{ftag(d['ratio-per-omic'])}{c}__seed{seed}"
        if axis == "nomics":
            return f"{base}__k{d['k-per-omic']}__nb{d['neighbor-size']}__emb{d['embedding-dims']}__omics-{omics_tag(d)}{c}__seed{seed}"
        if axis == "npatients":
            return f"{base}__k{d['k-per-omic']}__nb{d['neighbor-size']}__emb{d['embedding-dims']}__n{d['n-patients']}{c}__seed{seed}"
    if model == "pnet":
        base = f"pnet__hu{d['hidden-units']}__do{ftag(d['dropout'])}__wr{ftag(d['w-reg'])}"
        if axis == "missing":
            return f"{base}__k{d['k-per-omic']}__frac{ftag(d['include-non-intersection-frac'])}__seed{seed}"
        if axis == "ratio":
            return f"{base}__ratio{ftag(d['ratio-per-omic'])}__seed{seed}"
        if axis == "nomics":
            return f"{base}__k{d['k-per-omic']}__omics-{omics_tag(d)}__seed{seed}"
        if axis == "npatients":
            return f"{base}__k{d['k-per-omic']}__n{d['n-patients']}__seed{seed}"
    if model == "mofa":
        base = f"mofa__nl{d['n-latent']}__C{ftag(d['downstream-c'])}"
        if axis == "missing":
            return f"{base}__k{d['k-per-omic']}__frac{ftag(d['include-non-intersection-frac'])}__seed{seed}"
        if axis == "ratio":
            return f"{base}__ratio{ftag(d['ratio-per-omic'])}__seed{seed}"
        if axis == "nomics":
            return f"{base}__k{d['k-per-omic']}__omics-{omics_tag(d)}__seed{seed}"
        if axis == "npatients":
            return f"{base}__k{d['k-per-omic']}__n{d['n-patients']}__seed{seed}"
    raise ValueError(model)

def completed_set(model, ds, axis_out):
    done = set()
    for cls_dir in glob.glob(f"{ROOT}/{model}/results/{ds}/{axis_out}/*/*/"):
        with os.scandir(cls_dir) as it:
            for e in it:
                if e.is_dir():
                    done.add(TS.sub("", e.name))
    return done

grand = {}
for model in ["integrao", "pnet", "mofa"]:
    for ds in DATASETS:
        for axis_key, axis_out in AXES.items():
            f = f"{ROOT}/{model}/{model[:0]}experiments_{ds}_{axis_key}.txt".replace("//", "/")
            f = f"{ROOT}/{model}/experiments_{ds}_{axis_key}.txt"
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                continue
            done = completed_set(model, ds, axis_out)
            failed = []
            with open(f) as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    rn = run_name(model, axis_out, parse(line))
                    if rn not in done:
                        failed.append(line)
            if failed:
                out = f"{ROOT}/{model}/rerun_{ds}_{axis_key}.txt"
                with open(out, "w") as w:
                    w.write("\n".join(failed) + "\n")
                grand.setdefault(model, 0)
                grand[model] += len(failed)
                print(f"{model:9s} {ds:11s} {axis_out:10s} failed={len(failed):5d} done={len(done):5d}  -> {os.path.basename(out)}")

print("\nTOTAL failed to rerun by model:", grand, " total=", sum(grand.values()))
