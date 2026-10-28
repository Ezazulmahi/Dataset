r"""Step 1 - generate 5,000 cover images for training the universal-DCT steganalyser.

Covers come from the same two generators as the 400-pair evaluation corpus (unmodified
Stable Diffusion v1.5 and the fine-tuned LoRA model stored in F:\Results\unet.pt), with the
same sampler settings as F:\Results\validate.py (DDIM, 50 steps, no classifier-free guidance,
256x256, bfloat16). Prompts cycle through the 400-prompt bank; seeds start at 100000, so no
(prompt, seed) pair of the evaluation corpus (seeds 42..441) is reused.

Output: covers/ft_00000.png ... (2,500 fine-tuned) and covers/bs_00000.png ... (2,500 baseline)
Resumable: existing files are skipped.
"""
import sys, os, time, importlib.util
from pathlib import Path
import numpy as np, torch, cv2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "covers"; OUT.mkdir(exist_ok=True)
N_PER_MODEL = 2500
SEED0 = 100000
BATCH = 10

sys.path.insert(0, r"F:\Results"); sys.argv = ["validate.py"]
spec = importlib.util.spec_from_file_location("validate", r"F:\Results\validate.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
PROMPTS = V.PROMPTS
dev = torch.device("cuda"); dt = torch.bfloat16


@torch.no_grad()
def generate_batch(tok, enc, vae, unet, sched, prompts, seeds):
    tokens = tok(prompts, padding="max_length", max_length=tok.model_max_length, truncation=True,
                 return_tensors="pt").input_ids.to(dev)
    emb = enc(tokens).last_hidden_state.to(dtype=dt)
    g = torch.Generator(device=dev); lat = []
    for s in seeds:                      # one generator state per image, exactly as in validate.py
        g.manual_seed(int(s))
        lat.append(torch.randn(1, 4, V.RESOLUTION // 8, V.RESOLUTION // 8, device=dev, dtype=dt, generator=g))
    lat = torch.cat(lat) * sched.init_noise_sigma
    for t in sched.timesteps:
        lat = sched.step(unet(lat, t, encoder_hidden_states=emb).sample, t, lat).prev_sample
    img = vae.decode(lat / vae.config.scaling_factor).sample.float().cpu().permute(0, 2, 3, 1).numpy()
    return np.clip(img, -1.0, 1.0).astype(np.float32)


def run(finetuned, tag, check=False):
    tok, enc, vae, unet, sched = V.load_pipeline(dev, dt, finetuned=finetuned, finetuned_dir=Path(r"F:\Results"))
    sched.set_timesteps(V.DDIM_STEPS)
    if check:   # batching must reproduce the stored evaluation covers (prompt i, seed 42+i)
        imgs = generate_batch(tok, enc, vae, unet, sched, PROMPTS[:BATCH], [42 + i for i in range(BATCH)])
        ps = []
        for i, im in enumerate(imgs):
            ref = cv2.imread(rf"F:\Results\{'finetuned' if finetuned else 'baseline'}\{i:03d}.png").astype(float)
            mse = ((ref - V.float_rgb_to_uint8_bgr(im)) ** 2).mean(); ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
        print(f"[{tag}] batched sampler vs stored evaluation covers: PSNR min {min(ps):.1f} median {np.median(ps):.1f} dB", flush=True)
    todo = [k for k in range(N_PER_MODEL) if not (OUT / f"{tag}_{k:05d}.png").exists()]
    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        ks = todo[b:b + BATCH]
        imgs = generate_batch(tok, enc, vae, unet, sched, [PROMPTS[k % len(PROMPTS)] for k in ks], [SEED0 + k for k in ks])
        for k, im in zip(ks, imgs):
            cv2.imwrite(str(OUT / f"{tag}_{k:05d}.png"), V.float_rgb_to_uint8_bgr(im))
        if (b // BATCH) % 10 == 0:
            done = b + len(ks); el = time.time() - t0
            print(f"[{tag}] {done}/{len(todo)}  {el/done:.2f} s/img  eta {el/done*(len(todo)-done)/60:.0f} min", flush=True)
    del tok, enc, vae, unet, sched; torch.cuda.empty_cache()


if __name__ == "__main__":
    run(True, "ft", check=True)
    run(False, "bs", check=True)
    print("DONE", len(list(OUT.glob("*.png"))), flush=True)
