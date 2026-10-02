"""Recompute the two FID values of the manuscript with pytorch-fid (2,048-dimensional pool features)."""
from pytorch_fid.fid_score import calculate_fid_given_paths
if __name__ == "__main__":
    R = "F:/Results"
    a = calculate_fid_given_paths([R + "/baseline", R + "/finetuned"], batch_size=50, device="cuda", dims=2048, num_workers=0)
    b = calculate_fid_given_paths([R + "/finetuned", R + "/stego_finetuned"], batch_size=50, device="cuda", dims=2048, num_workers=0)
    print("FID baseline vs fine-tuned covers: %.3f (manuscript 271.04)" % a)
    print("FID fine-tuned covers vs stego:    %.3f (manuscript 4.88)" % b)
    open("fid_recompute.txt", "w").write("baseline_vs_finetuned %.4f\nfinetuned_vs_stego %.4f\n" % (a, b))
