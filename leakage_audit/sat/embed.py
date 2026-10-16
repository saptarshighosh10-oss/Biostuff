"""EuroSAT RGB -> frozen ResNet-50 (ImageNet V2) 2048-d embeddings + raw 64x64 pixels (uint8) for hashing. Run with ~/research-env python."""
import glob, os, numpy as np, torch, torchvision as tv
from PIL import Image
R = "data/external/eurosat_raw/x/EuroSAT_RGB"; O = "leakage_audit/sat"
files = sorted(glob.glob(f"{R}/*/*.jpg")); classes = sorted(os.listdir(R))
y = np.array([classes.index(f.split("/")[-2]) for f in files])
px = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]); print(px.shape)
m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity(); m.eval()
mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
E = []
with torch.no_grad():
    for i in range(0, len(px), 256):
        x = torch.from_numpy(px[i:i + 256]).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)
        E.append(m((x - mean) / std).numpy())
        if i % 2560 == 0: print(i, flush=True)
np.savez_compressed(f"{O}/eurosat.npz", emb=np.concatenate(E).astype(np.float32), px=px, y=y, files=np.array(files), classes=np.array(classes))
print("done")
