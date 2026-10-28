r"""Expansion of the evaluation corpus from 400 to 1,000 pairs, step 1: covers 400 to 999.

Generation is done with the authors' own code: load_pipeline() and generate_images() of
F:\Results\validate.py, with the adapter weights in F:\Results (unet.pt, vae.pt). Nothing in
F:\Results is modified. The 600 new prompts are 60 new texture subjects (none in the 40 subjects of
prompts.py) combined with the same ten modifiers. Prompt i is sampled with seed 42 + i, the rule of
the original 400, so the new seeds are 442 to 1041."""
import sys, json
from pathlib import Path
import torch, cv2
sys.path.insert(0, r"F:\Results")
import validate as V

HERE = Path(__file__).resolve().parent
SUBJECTS = ['zebra fur stripes', 'sheep wool fleece', 'fish scales', 'a peacock feather', 'turtle shell plates', 'crocodile skin',
            'butterfly wing scales', 'porcupine quills', 'a bird nest of twigs', 'a sea sponge surface',
            'fern leaves', 'cactus spines', 'a sunflower seed head', 'birch tree bark', 'dry autumn leaves on the ground',
            'a mushroom gill underside', 'wheat stalks in a field', 'tangled tree roots', 'palm leaf veins', 'a pomegranate seed cluster',
            'cracked ice on a frozen lake', 'gravel on a path', 'a layered sandstone cliff', 'a marble slab with veins', 'a slate roof',
            'wet beach pebbles', 'cracked asphalt', 'a concrete wall with pores', 'salt crystals', 'basalt columns',
            'burlap sack cloth', 'a tweed jacket weave', 'a braided rope coil', 'a wicker chair seat', 'embroidered cotton fabric',
            'corduroy fabric', 'a fishing net', 'a persian carpet pattern', 'a straw hat weave', 'velvet fabric folds',
            'a rusted chain-link fence', 'a stack of firewood logs', 'a thatched straw roof', 'weathered wooden planks', 'a dry stone wall',
            'a hammered copper sheet', 'a perforated metal grille', 'stacked roof shingles', 'a bamboo fence', 'an old cork board',
            'orange peel surface', 'the crumb of a sliced loaf of bread', 'coffee beans', 'a pile of rice grains', 'walnut shells',
            'broccoli florets', 'strawberry skin with seeds', 'a corn cob with kernels', 'cauliflower surface', 'dried lentils']
MODS = [p.split(", ", 1)[1] for p in V.PROMPTS[:10]]
OLD = {p[len("a photograph of "):].split(", ")[0] for p in V.PROMPTS}
assert len(SUBJECTS) == 60 == len(set(SUBJECTS)) and not (set(SUBJECTS) & OLD)
NEW_PROMPTS = [f"a photograph of {s}, {m}" for s in SUBJECTS for m in MODS]
FIRST = len(V.PROMPTS)                                    # 400

if __name__ == "__main__":
    (HERE / "prompts_400_999.txt").write_text("\n".join(NEW_PROMPTS), encoding="utf-8")
    json.dump({"subjects": SUBJECTS, "modifiers": MODS, "first_index": FIRST, "seed_rule": "42 + index"}, open(HERE / "prompts_400_999.json", "w"), indent=1)
    dev = torch.device("cuda"); dt = torch.bfloat16
    for finetuned, tag in ((True, "finetuned"), (False, "baseline")):
        out = HERE / "new600" / tag; out.mkdir(parents=True, exist_ok=True)
        if len(list(out.glob("*.png"))) == 600:           # resume: this set is already complete
            print(tag, "already complete", flush=True); continue
        tok, enc, vae, unet, sched = V.load_pipeline(dev, dt, finetuned=finetuned, finetuned_dir=Path(r"F:\Results"))
        if tag == "finetuned":     # the authors' function must reproduce a stored cover before it is used for new ones
            chk = V.float_rgb_to_uint8_bgr(V.generate_images(tok, enc, vae, unet, sched, dev, dt, V.PROMPTS[:2], seed=42)[1])
            ref = cv2.imread(r"F:\Results\finetuned\001.png"); d = int(abs(chk.astype(int) - ref.astype(int)).max())
            print("check against stored finetuned/001.png: max abs difference", d, flush=True)
        for b in range(0, 600, 50):
            if all((out / f"{FIRST + b + k:03d}.png").exists() for k in range(50)):
                continue
            imgs = V.generate_images(tok, enc, vae, unet, sched, dev, dt, NEW_PROMPTS[b:b + 50], seed=42 + FIRST + b)
            for k, im in enumerate(imgs):
                cv2.imwrite(str(out / f"{FIRST + b + k:03d}.png"), V.float_rgb_to_uint8_bgr(im))
            print(tag, b + 50, flush=True)
        del tok, enc, vae, unet, sched; torch.cuda.empty_cache()
    print("DONE", flush=True)
