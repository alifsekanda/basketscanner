"""SigLIP method (no VLM): crops are matched against the database photos (image search) and
the product names (text), and the best combined score wins."""
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from svlm.embed import _to_pil, pad_to_square


class SiglipEmbedder:
    """Same embed() interface as embed.Embedder, so build_index / ItemIndex work unchanged,
    plus product-name (text) embeddings."""

    def __init__(self, model_id, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        # SigLIP's own image-text match probability is sigmoid(cosine * scale + bias)
        self.logit_scale = self.model.logit_scale.exp().item()
        self.logit_bias = self.model.logit_bias.item()
        self._names = {}  # product name -> text vector, computed once

    @torch.inference_mode()
    def embed(self, images, batch_size=32):
        """images: file paths, PIL images or RGB arrays. Returns (N, D) L2-normalized float32."""
        out = []
        for i in range(0, len(images), batch_size):
            # The processor squashes images to a square, which distorts thin items: pad first
            batch = [pad_to_square(_to_pil(im)) for im in images[i:i + batch_size]]
            pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"]
            feats = self.model.get_image_features(pixel_values=pixels.to(self.device, self.model.dtype))
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.inference_mode()
    def name_vectors(self, names):
        """Text vectors for "This is a photo of <name>." (folder name, _ and - read as spaces)."""
        missing = [n for n in names if n not in self._names]
        if missing:
            texts = [f"this is a photo of {n.replace('_', ' ').replace('-', ' ')}." for n in missing]
            # SigLIP was trained on lowercase text padded to 64 tokens
            inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                    return_tensors="pt").to(self.device)
            feats = torch.nn.functional.normalize(self.model.get_text_features(**inputs).float(), dim=-1)
            self._names.update(zip(missing, feats.cpu().numpy()))
        return np.stack([self._names[n] for n in names])


class SiglipMatcher:
    """Stands in for the VLM: same identify() interface as vlm.ItemRecognizer. Each crop's
    candidates (closest products by image) are re-scored with how well the crop matches their name."""
    step_name = "name match"

    def __init__(self, embedder, text_weight):
        self.embedder, self.text_weight = embedder, text_weight

    def identify(self, crops, candidates):
        if not crops:
            return []
        out = []
        for vec, cands in zip(self.embedder.embed(crops), candidates):
            names = [c["item"] for c in cands]
            cos = self.embedder.name_vectors(names) @ vec
            text = 1 / (1 + np.exp(-(cos * self.embedder.logit_scale + self.embedder.logit_bias)))
            image = np.array([c["score"] for c in cands])
            score = (1 - self.text_weight) * image + self.text_weight * text
            best = int(score.argmax())
            out.append({"item": names[best], "confidence": round(float(score[best]), 3),
                        "reason": f"image {image[best]:.2f}, name {text[best]:.2f}", "raw": ""})
        return out
