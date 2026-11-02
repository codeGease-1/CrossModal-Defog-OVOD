"""Pinned local SegFormer weights. No downloads or training at inference."""
import json
import torch
from torch import nn
from torch.nn import functional as F


class FrozenSemantics(nn.Module):
    def __init__(self, model_path, groups):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        from pathlib import Path
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            str(model_path), local_files_only=True, use_safetensors=True).eval()
        self.model.requires_grad_(False)
        self.config = json.loads((Path(model_path)/'preprocessor_config.json').read_text())
        labels = self.model.config.id2label
        self.indices = [[i for i, name in labels.items() if name in group] for group in groups]
        if any(not values for values in self.indices):
            raise ValueError('Semantic labels not found in frozen model')
        self.register_buffer('mean', torch.tensor(self.config['image_mean'])[None, :, None, None])
        self.register_buffer('std', torch.tensor(self.config['image_std'])[None, :, None, None])

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('Expected finite display RGB in [0,1]')
        size = self.config['size']
        target_size = (size, size) if isinstance(size, int) else (size['height'], size['width'])
        x = F.interpolate(rgb, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        logits = self.model(pixel_values=(x-self.mean)/self.std).logits
        probabilities = F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False).softmax(1)
        masks = [probabilities[:, ids].sum(1, keepdim=True) for ids in self.indices]
        other = (1-torch.cat(masks, 1).sum(1, keepdim=True)).clamp_min(0)
        result = torch.cat([other]+masks, 1)
        return result / result.sum(1, keepdim=True)
