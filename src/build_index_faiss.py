import os, numpy as np, faiss, open_clip, torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model.eval()
    return model, preprocess

def encode_image(model, preprocess, path):
    x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        v = model.encode_image(x).float().cpu().numpy()
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v[0]

def main():
    indir = "data/samples"
    model, preprocess = load_model()
    vecs, names = [], []
    for fname in tqdm(os.listdir(indir)):
        path = os.path.join(indir, fname)
        if not os.path.isfile(path): continue
        try:
            vecs.append(encode_image(model, preprocess, path))
            names.append(fname)
        except Exception as e:
            print("skip:", fname, e)

    if not vecs:
        print("no images"); return

    xb = np.stack(vecs).astype('float32')
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.write_index(index, "outputs/faiss.index")
    np.save("outputs/names.npy", np.array(names))
    print("built:", len(names))

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
