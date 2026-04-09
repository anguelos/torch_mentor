import os
import sys
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import fargv
import mentor

MODEL_ID = "google/mobilenet_v2_1.0_224"
CACHE    = "./tmp/mobilenetv2.hf"

# Download once, save locally; load from cache on subsequent runs

def main():
    args, _ = fargv.parse({
        "img": [],
        "hf_cache": CACHE,
        "model_id": MODEL_ID,
    })

    if not os.path.exists(args.hf_cache):
        model     = AutoModelForImageClassification.from_pretrained(args.model_id)
        processor = AutoImageProcessor.from_pretrained(args.model_id)
        os.makedirs(args.hf_cache, exist_ok=True)
        model.save_pretrained(args.hf_cache)
        processor.save_pretrained(args.hf_cache)
    else:
        model     = AutoModelForImageClassification.from_pretrained(args.hf_cache)
        processor = AutoImageProcessor.from_pretrained(args.hf_cache)

    model = mentor.wrap_as_mentee(model)
    model.eval()
    with torch.no_grad():
        for img_path in args.img:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            logits = model(**inputs).logits
            probs   = logits.softmax(-1)[0]
            top5    = probs.topk(5)
            for score, idx in zip(top5.values, top5.indices):
                print(f"{score:.3f}  {model.config.id2label[idx.item()]},   ", end="")
            print()
    model.save(f"{args.hf_cache}.mentor.pt")


if __name__ == "__main__":
    main()