"""
Computes CLIP cosine similarity between finetuned cover images and their
matching stego images using the mapping stored in comparison_results_pilot.xlsx.

Requires:
    pip install torch transformers pillow pandas openpyxl
"""

import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FINETUNED_DIR = r"F:\Results\finetunecover"
STEGO_DIR = r"F:\Results\stego_pilot"
EXCEL_PATH = r"F:\Results\comparison_results_pilot.xlsx"

MODEL_NAME = "openai/clip-vit-base-patch32"

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def load_image_map(folder):
    """Map filename (without extension) -> full path."""
    mapping = {}

    for fname in os.listdir(folder):
        if fname.lower().endswith(VALID_EXTS):
            mapping[os.path.splitext(fname)[0]] = os.path.join(folder, fname)

    return mapping


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model on {device}...")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # ---------------------------------------------------------
    # Read mapping from Excel
    # ---------------------------------------------------------
    df = pd.read_excel(EXCEL_PATH, sheet_name="per_image_results")

    # source_finetuned_file -> stego image
    mapping = dict(zip(df["source_finetuned_file"], df["image"]))

    finetuned_map = load_image_map(FINETUNED_DIR)
    stego_map = load_image_map(STEGO_DIR)

    pairs = []

    for finetuned_file, stego_file in mapping.items():

        finetuned_key = os.path.splitext(finetuned_file)[0]
        stego_key = os.path.splitext(stego_file)[0]

        finetuned_path = finetuned_map.get(finetuned_key)
        stego_path = stego_map.get(stego_key)

        if finetuned_path is None:
            print(f"Missing finetuned image: {finetuned_file}")
            continue

        if stego_path is None:
            print(f"Missing stego image: {stego_file}")
            continue

        pairs.append((finetuned_file, stego_file, finetuned_path, stego_path))

    if len(pairs) == 0:
        print("No valid pairs found.")
        return

    print(f"\nFound {len(pairs)} matched image pairs.\n")

    scores = []

    for finetuned_name, stego_name, finetuned_path, stego_path in pairs:

        img1 = Image.open(finetuned_path).convert("RGB")
        img2 = Image.open(stego_path).convert("RGB")

        inputs = processor(
            images=[img1, img2],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        # Handle different transformers versions
        if not isinstance(features, torch.Tensor):

            if hasattr(features, "image_embeds"):
                features = features.image_embeds

            elif hasattr(features, "pooler_output"):
                features = features.pooler_output

            else:
                raise RuntimeError(
                    f"Unsupported output type: {type(features)}"
                )

        features = features / features.norm(dim=-1, keepdim=True)

        cosine_sim = torch.nn.functional.cosine_similarity(
            features[0].unsqueeze(0),
            features[1].unsqueeze(0)
        ).item()

        scores.append(cosine_sim)

        print(f"{finetuned_name} --> {stego_name} : {cosine_sim:.6f}")

    mean_score = sum(scores) / len(scores)

    print("\n===================================")
    print(f"Mean CLIP cosine similarity : {mean_score:.6f}")
    print(f"Images compared             : {len(scores)}")
    print("===================================")


if __name__ == "__main__":
    main()