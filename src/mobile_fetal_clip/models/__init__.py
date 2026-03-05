"""Model definitions and registration for MobileCLIP2 fetal variants."""

from mobile_fetal_clip.models.mci_registry import register_all_mci_models
from mobile_fetal_clip.models.factory import create_fetal_clip_model, get_tokenizer, load_pretrained_weights

# Register MCI timm models on import
register_all_mci_models()
