"""Qwen2.5-VL picks which database item a crop is, among retrieved candidates."""
import re
import string
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from svlm import load_rgb

PROMPT = """You are identifying a product taken out of a shopping basket.
The first image is the QUERY item. The other images are CANDIDATE items from the product database, each labelled with a letter and its name.

Decide which candidate is the SAME product as the query. Compare shape, colour, packaging text, brand logo and size.
The query may be rotated, partly occluded or photographed at a different angle.
If none of the candidates is the same product, answer "none".

Reply with JSON only, no other text:
{{"choice": one of {choices} or "none", "confidence": number between 0 and 1, "reason": "short reason"}}"""

BATCH_SIZE = 8  # crops per generate call; caps GPU memory for baskets with many items


class ItemRecognizer:
    def __init__(self, model_id, max_pixels=448 * 448, max_new_tokens=128, device="cuda"):
        # The generic loader picks the right class for Qwen2.5-VL and Qwen3-VL
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"  # batched generation needs left padding
        # qwen-vl-utils must resize images to the model's patch grid: 14 px (Qwen2.5-VL), 16 px (Qwen3-VL)
        self.patch_size = self.processor.image_processor.patch_size
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

    def _image(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return {"type": "image", "image": img, "max_pixels": self.max_pixels}

    @torch.inference_mode()
    def identify(self, crops, candidates):
        """crops: RGB arrays. candidates: per crop, [{"item", "path", "score"}] from ItemIndex.search.
        Crops are answered in batches, which is much faster than one generate call per crop.
        Returns one {"item", "confidence", "reason", "raw"} per crop; item is None when unknown."""
        results = []
        for start in range(0, len(crops), BATCH_SIZE):
            batch = list(zip(crops[start:start + BATCH_SIZE], candidates[start:start + BATCH_SIZE]))
            messages, letters = [], []
            for crop, cands in batch:
                # A list, not a string: `"" in "ABCDE"` is True, `"" in ["A", ...]` is not.
                ls = list(string.ascii_uppercase[:len(cands)])
                content = [{"type": "text", "text": "QUERY:"}, self._image(crop)]
                for letter, cand in zip(ls, cands):
                    content.append({"type": "text", "text": f"{letter}) {cand['item']}:"})
                    content.append(self._image(_load_ref(cand["path"])))
                content.append({"type": "text", "text": PROMPT.format(
                    choices=", ".join(f'"{l}"' for l in ls))})
                messages.append([{"role": "user", "content": content}])
                letters.append(ls)

            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in messages]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=self.patch_size)
            # do_resize=False: qwen-vl-utils has already resized the images to fit the patch grid
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False,
                                    padding=True, return_tensors="pt").to(self.model.device)
            # The answer gives "choice" and "confidence" before "reason", so stopping as soon as
            # "reason" starts keeps the decision and skips most of the generation time.
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      stop_strings=['"reason"'], tokenizer=self.processor.tokenizer)
            raws = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True)

            for raw, ls, (_, cands) in zip(raws, letters, batch):
                choice, confidence = parse_answer(raw)
                item = cands[ls.index(choice)]["item"] if choice in ls else None
                results.append({"item": item, "confidence": confidence, "reason": "", "raw": raw.strip()})
        return results


@lru_cache(maxsize=256)
def _load_ref(path):
    """Reference images repeat across crops and photos; load each one once."""
    return load_rgb(path)


def parse_answer(raw):
    """Returns (choice, confidence) from the answer, which is cut off before "reason", so it is
    never complete JSON. choice is an uppercase letter, "none" or None."""
    m = re.search(r'"?choice"?\s*:\s*"?([A-Za-z]+)"?', raw)
    if not m:
        return None, 0.0
    choice = "none" if m.group(1).lower() == "none" else m.group(1).upper()
    c = re.search(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', raw)
    return choice, float(c.group(1)) if c else 0.0
