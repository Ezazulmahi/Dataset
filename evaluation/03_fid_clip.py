"""
03_fid_clip.py  -  FID and CLIP similarity between the fine-tuned covers and their v5
stego images (as in the manuscript: 2,048-d Inception features; CLIP ViT-B/32 image
encoder), using the authors' functions from fid.py unchanged.  Runs on the CPU by
default so it does not compete with GPU training (pass "cuda" to use the GPU).
FID between baseline and fine-tuned covers does not depend on the embedder and is
not recomputed (238.19 on 1,000 pairs, report_1000.txt).
Output: results/fid_clip_v5.csv, results/clip_per_image_v5.csv
"""
import os, sys, importlib.util
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
spec = importlib.util.spec_from_file_location("fid", r"F:\scirep results\09_expansion_1000\authors_scripts_paths_changed\fid.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
ft = r"F:\scirep results\09_expansion_1000\corpus1000\finetuned"
st = os.path.join(HERE, "stego_finetuned")
assert len(os.listdir(st)) == 1000, f"{len(os.listdir(st))} stego images; wait for 01_embed_normal_1000.py"
from pytorch_fid.fid_score import calculate_fid_given_paths
# same call as the authors' compute_fid (batch 50, 2,048-d features), but with the
# data loading in the main process: worker processes crashed on this machine.
fid = calculate_fid_given_paths([ft, st], batch_size=50, device=dev, dims=2048, num_workers=0)
clip = F.compute_clip_similarity(ft, st, device=dev)
clip.to_csv(os.path.join(HERE, "results", "clip_per_image_v5.csv"), index=False)
col = [c for c in clip.columns if "clip" in c.lower() or "cos" in c.lower()][0]
s = pd.DataFrame([dict(FID_finetuned_vs_stego_v5=fid, CLIP_mean=clip[col].mean(), CLIP_sd=clip[col].std(),
                       CLIP_min=clip[col].min(), n=len(clip))])
s.to_csv(os.path.join(HERE, "results", "fid_clip_v5.csv"), index=False); print(s.to_string(index=False))
