"""Test 2: are the baseline covers weak only because of the sampling settings?

The corpus was sampled without classifier-free guidance (CFG) at 256x256. Here both models are sampled the usual way:
  cfg256 : CFG scale 7.5 at 256x256, prompts 0, 5, ..., 995 (200 prompts), seed 42 + index as in the corpus
  cfg512 : CFG scale 7.5 at 512x512 (the native size), prompts 0, 10, ..., 990 (100 prompts); statistics are
           computed on the 512 image itself and on a 256 downscale (area interpolation)
Same 50-step DDIM sampler, same prompt and seed per pair. Before anything else the script regenerates corpus
image 000 of both models with the corpus settings and checks it is identical to the stored cover.
Resumable: images are saved as PNG and results/test2_generation.csv gets a row per image at once; a STOP file
stops it after the current image."""
import os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
import common as C

OUT = os.path.join(C.HERE, "results", "test2_generation.csv")
IMG = Path(C.HERE) / "test2_images"
CFG = 7.5

sys.path.insert(0, r"F:\Results"); sys.argv = ["validate.py"]
spec = importlib.util.spec_from_file_location("validate", r"F:\Results\validate.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
PROMPTS = list(V.PROMPTS) + Path(r"F:\scirep results\09_expansion_1000\prompts_400_999.txt").read_text(encoding="utf-8").splitlines()
assert len(PROMPTS) == 1000
dev = torch.device("cuda"); dt = torch.bfloat16


@torch.no_grad()
def generate(tok, enc, vae, unet, sched, prompt, seed, res, cfg):
    def emb(p):
        ids = tok([p], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(dev)
        return enc(ids).last_hidden_state.to(dtype=dt)
    cond = emb(prompt)
    g = torch.Generator(device=dev); g.manual_seed(int(seed))
    lat = torch.randn(1, 4, res // 8, res // 8, device=dev, dtype=dt, generator=g) * sched.init_noise_sigma
    if cfg is None:      # corpus setting: one conditional pass per step, exactly as validate.py
        for t in sched.timesteps:
            lat = sched.step(unet(lat, t, encoder_hidden_states=cond).sample, t, lat).prev_sample
    else:
        both = torch.cat([emb(""), cond])
        for t in sched.timesteps:
            e_u, e_c = unet(torch.cat([lat, lat]), t, encoder_hidden_states=both).sample.chunk(2)
            lat = sched.step(e_u + cfg * (e_c - e_u), t, lat).prev_sample
    img = vae.decode(lat / vae.config.scaling_factor).sample[0].float().cpu().permute(1, 2, 0).numpy()
    return V.float_rgb_to_uint8_bgr(np.clip(img, -1.0, 1.0).astype(np.float32))


COLS = ["key", "model", "setting", "index", "res", "identical_to_corpus", "e_bar", "K", "capacity_bits", "capacity_bpp",
        "hf_share", "slope", "down256_e_bar", "down256_K", "down256_capacity_bits", "down256_capacity_bpp"]


def save(row):
    pd.DataFrame([row]).reindex(columns=COLS).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)


def main():
    IMG.mkdir(exist_ok=True); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    D = C.load_v5()
    done = set(pd.read_csv(OUT).key) if os.path.exists(OUT) else set()
    jobs = [("cfg256", i, 256) for i in range(0, 1000, 5)] + [("cfg512", i, 512) for i in range(0, 1000, 10)]
    for finetuned, model in ((False, "baseline"), (True, "finetuned")):
        todo = [j for j in jobs if f"{model}:{j[0]}:{j[1]}" not in done]
        check_key = f"{model}:check:0"
        if not todo and check_key in done:
            continue
        tok, enc, vae, unet, sched = V.load_pipeline(dev, dt, finetuned=finetuned, finetuned_dir=Path(r"F:\Results"))
        sched.set_timesteps(V.DDIM_STEPS)
        if check_key not in done:
            again = generate(tok, enc, vae, unet, sched, PROMPTS[0], 42, 256, None)
            stored = cv2.imread(os.path.join(C.CORP, model, "000.png"))
            same = bool(np.array_equal(again, stored))
            print(f"{model}: corpus image 000 regenerated identically: {same}", flush=True)
            save(dict(key=check_key, model=model, setting="check", index=0, res=256, identical_to_corpus=same))
        for k, (setting, i, res) in enumerate(todo, 1):
            if os.path.exists(C.STOP):
                print("STOP file found: everything finished so far is saved, exiting", flush=True); return False
            img = generate(tok, enc, vae, unet, sched, PROMPTS[i], 42 + i, res, CFG)
            cv2.imwrite(str(IMG / f"{model}_{setting}_{i:03d}.png"), img)
            row = dict(key=f"{model}:{setting}:{i}", model=model, setting=setting, index=i, res=res,
                       **C.cover_stats(D, img), **C.spectrum_stats(img))
            if res == 512:
                small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                s = C.cover_stats(D, small)
                row.update({f"down256_{a}": b for a, b in s.items()})
            save(row)
            print(f"  {model} {setting} {i:03d} saved ({k}/{len(todo)})", flush=True)
        del tok, enc, vae, unet, sched; torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
