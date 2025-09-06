import argparse, yaml, os
import numpy as np
from PIL import Image
import open_clip
import torch
from torchvision import transforms
from tqdm import tqdm

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    return model, preprocess, tokenizer

def make_prompts(label_dict):
    # 카테고리별 프롬프트 생성 (간단 버전)
    prompts = {}
    for key, values in label_dict.items():
        prompts[key] = [f"a photo of {v} clothing" for v in values]
    return prompts

def encode_texts(model, tokenizer, prompts):
    with torch.no_grad():
        text_embeds = {
            k: model.encode_text(tokenizer(v)).float() for k, v in prompts.items()
        }
    for k in text_embeds:
        text_embeds[k] /= text_embeds[k].norm(dim=-1, keepdim=True) + 1e-9
    return text_embeds

def encode_image(model, preprocess, path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(img).float()
    emb /= emb.norm(dim=-1, keepdim=True) + 1e-9
    return emb.squeeze(0).cpu().numpy()

def predict_multilabel(img_emb, text_embeds, topk=1):
    # 각 속성군에서 가장 유사한 라벨 선택(top-1)
    out = {}
    for k, ten in text_embeds.items():
        sims = (ten @ torch.tensor(img_emb)).squeeze(-1)
        topv, topi = torch.topk(sims, k=topk)
        out[k] = [i.item() for i in topi]
    return out

def main(args):
    with open(args.labels, "r", encoding="utf-8") as f:
        label_dict = yaml.safe_load(f)

    model, preprocess, tokenizer = load_model()
    prompts = make_prompts(label_dict)
    text_embeds = encode_texts(model, tokenizer, prompts)

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for fname in tqdm(os.listdir(args.indir)):
        path = os.path.join(args.indir, fname)
        if not os.path.isfile(path): continue
        try:
            img_emb = encode_image(model, preprocess, path)
            pred_idx = predict_multilabel(img_emb, text_embeds, topk=1)
            row = {"file": fname}
            for k, idxs in pred_idx.items():
                row[k] = list(label_dict[k].values()) if isinstance(label_dict[k], dict) else label_dict[k]
                labels = label_dict[k]
                row[k] = labels[idxs[0]]
            results.append(row)
        except Exception as e:
            print("skip:", fname, e)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.outdir, "tags.csv"), index=False)
    print("saved:", os.path.join(args.outdir, "tags.csv"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/samples")
    ap.add_argument("--labels", default="configs/labels.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    main(args)
