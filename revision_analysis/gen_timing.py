r"""Time cover generation with the authors' validate.py code path and check that the stored
evaluation covers are reproduced by the saved unet.pt (seed 42 + prompt index)."""
import sys, time, importlib.util, numpy as np, torch, cv2
from pathlib import Path
sys.path.insert(0, r"F:\Results"); sys.argv = ["validate.py"]
spec = importlib.util.spec_from_file_location("validate", r"F:\Results\validate.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
dev = torch.device("cuda"); dt = torch.bfloat16; n = 12
prompts = V.PROMPTS[:n]
def to_bgr(img):
    return cv2.cvtColor(np.clip(np.rint((img + 1) * 127.5), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
for ft, folder in ((False, "baseline"), (True, "finetuned")):
    torch.cuda.reset_peak_memory_stats()
    tok, enc, vae, unet, sched = V.load_pipeline(dev, dt, finetuned=ft, finetuned_dir=Path(r"F:\Results"))
    if ft:
        print("trainable-shaped LoRA params:", sum(p.numel() for k, p in unet.named_parameters() if "lora_" in k))
    V.generate_images(tok, enc, vae, unet, sched, dev, dt, prompts=prompts[:1], seed=42)   # warm-up
    torch.cuda.synchronize(); t = time.perf_counter()
    imgs = V.generate_images(tok, enc, vae, unet, sched, dev, dt, prompts=prompts, seed=42)
    torch.cuda.synchronize(); el = (time.perf_counter() - t) / n
    ps = []
    for i, im in enumerate(imgs):
        ref = cv2.imread(rf"F:\Results\{folder}\{i:03d}.png"); g = to_bgr(im)
        mse = ((ref.astype(float) - g) ** 2).mean(); ps.append(99 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
        cv2.imwrite(f"regen_{folder}_{i:03d}.png", g)
    print(f"{folder}: {el:.2f} s/image (50 DDIM steps, 256x256), peak GPU mem {torch.cuda.max_memory_allocated()/2**20:.0f} MiB, "
          f"PSNR regenerated-vs-stored: min {min(ps):.1f} median {np.median(ps):.1f} max {max(ps):.1f} dB")
    del tok, enc, vae, unet, sched; torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0))
