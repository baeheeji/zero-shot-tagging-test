import sys, numpy as np, faiss, open_clip, torch
from PIL import Image

def load():
    index = faiss.read_index("outputs/faiss.index")
    names = np.load("outputs/names.npy", allow_pickle=True)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()
    return index, names, model, preprocess

def encode(model, preprocess, path):
    x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        v = model.encode_image(x).float().cpu().numpy()
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v.astype('float32')

def main(img_path, k=5):
    index, names, model, preprocess = load()
    q = encode(model, preprocess, img_path)
    D, I = index.search(q, k)
    for d, i in zip(D[0], I[0]):
        print(f"{names[i]}  cos={d:.4f}")

if __name__ == "__main__":
    main(sys.argv[1], k=5)
