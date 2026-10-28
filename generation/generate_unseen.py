r"""Unseen-prompt check, step 1: generate 200 baseline / fine-tuned cover pairs from 20 subjects that
are NOT in the 400-prompt bank and are not texture close-ups. Same generators and sampler settings
as the evaluation corpus; seeds 200000 + index, used for no other image."""
import sys, importlib.util
from pathlib import Path
import numpy as np, torch, cv2

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen", r"F:\Steganalyzer_UniversalDCT\01_generate_covers.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
V = G.V

SUBJECTS = ["a red apple on a wooden table", "a city street at night", "a mountain lake", "a bowl of fruit", "a bicycle leaning against a wall",
            "a sailing boat on the sea", "a bookshelf full of books", "a cup of coffee on a desk", "a dog sitting on grass", "a wooden chair in an empty room",
            "a vase of flowers", "a lighthouse on a cliff", "a forest path in autumn", "a kitchen counter with utensils", "a parked car on a street",
            "a bridge over a river", "a child's toy blocks", "a pair of shoes on a doormat", "a market stall with vegetables", "a snowy village rooftop"]
MODS = [p.split(", ", 1)[1] for p in V.PROMPTS[:10]]
PROMPTS = [f"a photograph of {s}, {m}" for s in SUBJECTS for m in MODS]
assert len(PROMPTS) == 200 and not (set(PROMPTS) & set(V.PROMPTS))

if __name__ == "__main__":
    (HERE / "prompts_unseen.txt").write_text("\n".join(PROMPTS), encoding="utf-8")
    for finetuned, tag in ((True, "finetuned"), (False, "baseline")):
        out = HERE / tag; out.mkdir(exist_ok=True)
        tok, enc, vae, unet, sched = V.load_pipeline(G.dev, G.dt, finetuned=finetuned, finetuned_dir=Path(r"F:\Results"))
        sched.set_timesteps(V.DDIM_STEPS)
        for b in range(0, 200, 10):
            imgs = G.generate_batch(tok, enc, vae, unet, sched, PROMPTS[b:b + 10], [200000 + k for k in range(b, b + 10)])
            for k, im in zip(range(b, b + 10), imgs):
                cv2.imwrite(str(out / f"{k:03d}.png"), V.float_rgb_to_uint8_bgr(im))
        del tok, enc, vae, unet, sched; torch.cuda.empty_cache()
        print(tag, "done", len(list(out.glob('*.png'))), flush=True)
