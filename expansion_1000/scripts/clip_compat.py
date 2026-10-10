"""Compatibility shim for transformers >= 5: CLIPModel.get_image_features() now returns a
BaseModelOutputWithPooling whose pooler_output is the projected image embedding that older versions
returned directly as a tensor. Importing this module restores the old return type, so the authors'
CLIP code in comparison.py and fid.py runs unchanged."""
import transformers
_orig = transformers.CLIPModel.get_image_features
def _get_image_features(self, *a, **k):
    out = _orig(self, *a, **k)
    return out if hasattr(out, "norm") else out.pooler_output
transformers.CLIPModel.get_image_features = _get_image_features
