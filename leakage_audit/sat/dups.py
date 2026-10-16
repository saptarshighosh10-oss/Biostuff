import glob, os, hashlib, numpy as np, collections
from PIL import Image
R="data/external/eurosat_raw/x/EuroSAT_RGB"
F=sorted(glob.glob(f"{R}/*/*.jpg")); C=[f.split("/")[-2] for f in F]
ex=collections.defaultdict(list); dh=collections.defaultdict(list)
for i,f in enumerate(F):
    im=Image.open(f).convert("RGB"); a=np.asarray(im)
    ex[hashlib.md5(a.tobytes()).hexdigest()].append(i)
    g=np.asarray(im.convert("L").resize((9,8),Image.LANCZOS),dtype=np.int16)
    dh[(g[:,1:]>g[:,:-1]).tobytes()].append(i)
def rep(name,d):
    gs=[v for v in d.values() if len(v)>1]
    inv=sum(len(v) for v in gs); dup=sum(len(v)-1 for v in gs)
    conf=sum(1 for v in gs if len({C[i] for i in v})>1)
    print(f"{name}: groups={len(gs)} tiles_in_groups={inv} redundant={dup} ({dup/len(F):.2%}) label_conflict_groups={conf}")
    return gs
rep("exact pixel",ex); g=rep("dHash64 identical",dh)
print("tiles",len(F),"names unique",len({os.path.basename(f) for f in F}))
print("--- group inspection")
rows=[]
for v in g:
    A=np.stack([np.asarray(Image.open(F[i]).convert("RGB"),dtype=np.float32) for i in v])
    mad=max(np.abs(A[a]-A[b]).mean() for a in range(len(v)) for b in range(a+1,len(v)))
    rows.append((len(v),C[v[0]],mad,A.std(axis=(1,2,3)).mean(),v))
rows.sort(key=lambda r:r[2])
for n,c,m,s,v in rows: print(f"{n:3d} {c:20s} maxMAD={m:6.2f} tileStd={s:5.1f}")
print("by class:",collections.Counter(r[1] for r in rows))
print("MAD<2 (near-identical):",sum(r[2]<2 for r in rows),"MAD<10:",sum(r[2]<10 for r in rows),"of",len(rows))
import json; json.dump([[F[i] for i in r[4]] for r in rows],open("leakage_audit/sat/dup_groups.json","w"))
