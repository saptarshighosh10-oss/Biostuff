You are fine-tuning a protein language model (a neural net) to predict antibody polyreactivity,
on Google Colab with a GPU. Assume I'm a beginner. Be rigorously honest: report the REAL
held-out number, never fabricate, and if it doesn't beat the baseline, say so plainly. A null
result is a valid, useful answer.

## The question we're testing
Our current model (gradient-boosted trees on frozen features) hits ~0.57 AUC on DIVERSE
antibodies for polyreactivity — a wall. We now have 373k library sequences, finally enough to
fine-tune a neural net. Does fine-tuning a PLM beat 0.57 on antibodies it never trained on?

## Data
Upload ~/Downloads/antibody_sources/polyreactivity_dataset.csv
(columns: source, vh, vl, label, label_type). Sources:
- TRAIN on the LIBRARY sources: emi, igg, iso, tessier_lib1, tessier_lib2 (~370k rows).
- HELD-OUT TEST (never train on these): aims_mouse, aims_flu, shehata2019 (the diverse sets).
Binarize labels: library and aims_mouse labels are already 0/1; aims_flu is ordinal (0–6), so use
strictly above its within-source median as 1; for shehata2019 (continuous PSR) use "bad" = top 15%
highest PSR = 1, else 0. STRICT: drop any training VH that also appears in a holdout.

## Steps (paste each Colab cell)
# CELL 1 — GPU runtime (Runtime > Change runtime type > T4 GPU), then:
!pip -q install "transformers>=4.44" datasets accelerate scikit-learn

# CELL 2 — upload
from google.colab import files; files.upload()   # polyreactivity_dataset.csv

# CELL 3 — fine-tune ESM-2 (35M, fits a T4) with a classification head on the VH sequence
```python
import csv, numpy as np, torch
from collections import Counter, defaultdict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import roc_auc_score

LIB={"emi","igg","iso","tessier_lib1","tessier_lib2"}; HOLD={"aims_mouse","aims_flu","shehata2019"}
rows=[r for r in csv.DictReader(open("polyreactivity_dataset.csv")) if (r["vh"] or "").strip()]
# binarize
def lab(r):
    try: v=float(r["label"])
    except: return None
    return v
# shehata continuous -> top15% within source; others already 0/1
by_src=defaultdict(list)
for r in rows: by_src[r["source"]].append(r)
Y={}
for s,rs in by_src.items():
    vals=[lab(r) for r in rs]
    if s=="shehata2019":
        thr=np.quantile([v for v in vals if v is not None],0.85)
        for r,v in zip(rs,vals): Y[id(r)]=int(v>thr) if v is not None else None
    elif s=="aims_flu":
        thr=np.median([v for v in vals if v is not None])
        for r,v in zip(rs,vals): Y[id(r)]=int(v>thr) if v is not None else None
    else:
        for r,v in zip(rs,vals): Y[id(r)]=int(v) if v in (0.0,1.0) else None
hold_vh={(r["vh"] or "").strip().upper() for s in HOLD for r in by_src.get(s,[])}
raw_train=[r for s in LIB for r in by_src.get(s,[]) if Y[id(r)] is not None]
train=[r for r in raw_train if (r["vh"] or "").strip().upper() not in hold_vh]
print("library rows before strict holdout removal:", len(raw_train))
print("shared-VH training rows removed:", len(raw_train)-len(train))
# De-duplicate VH before sampling. If the same VH has conflicting labels, exclude it
# rather than silently choosing one label.
by_vh=defaultdict(list)
for r in train: by_vh[(r["vh"] or "").strip().upper()].append(r)
conflicts=sum(len(rs) for rs in by_vh.values() if len({Y[id(r)] for r in rs})>1)
train=[rs[0] for rs in by_vh.values() if len({Y[id(r)] for r in rs})==1]
print("conflicting-label duplicate rows removed:", conflicts)
print("unique library rows before cap:", len(train))
print("by source/class before cap:", {s: dict(Counter(Y[id(r)] for r in train if r["source"]==s)) for s in sorted(LIB)})
# cap for a first run so it finishes on a T4 (balanced); raise later if promising
import random; random.seed(42); random.shuffle(train)
pos=[r for r in train if Y[id(r)]==1][:40000]; neg=[r for r in train if Y[id(r)]==0][:40000]
train=pos+neg; random.shuffle(train)
print("train rows:",len(train))
print("by source/class after cap:", {s: dict(Counter(Y[id(r)] for r in train if r["source"]==s)) for s in sorted(LIB)})

tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model=AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t12_35M_UR50D",num_labels=2)
def enc(rs):
    e=tok([r["vh"][:200] for r in rs],padding=True,truncation=True,max_length=200,return_tensors="pt")
    e["labels"]=torch.tensor([Y[id(r)] for r in rs]); return e
from torch.utils.data import Dataset
class DS(Dataset):
    def __init__(s,rs): s.rs=rs
    def __len__(s): return len(s.rs)
    def __getitem__(s,i):
        r=s.rs[i]; e=tok(r["vh"][:200],truncation=True,max_length=200); e["labels"]=Y[id(r)]; return e
from transformers import DataCollatorWithPadding
args=TrainingArguments("out",per_device_train_batch_size=32,num_train_epochs=2,
    learning_rate=2e-5,fp16=True,logging_steps=200,report_to="none",save_strategy="no")
tr=Trainer(model,args,train_dataset=DS(train),data_collator=DataCollatorWithPadding(tok))
tr.train()

# EVALUATE on each held-out diverse source (never trained on)
model.eval()
for s in HOLD:
    rs=[r for r in by_src.get(s,[]) if Y[id(r)] is not None]
    if len(rs)<20: continue
    scores=[]
    for i in range(0,len(rs),64):
        b=rs[i:i+64]; e=tok([r["vh"][:200] for r in b],padding=True,truncation=True,max_length=200,return_tensors="pt").to(model.device)
        with torch.no_grad(): p=torch.softmax(model(**e).logits,-1)[:,1].cpu().numpy()
        scores+=list(p)
    y=[Y[id(r)] for r in rs]
    print(f"{s}: n={len(rs)} AUC={roc_auc_score(y,scores):.3f}  (tree baseline diverse ~0.57)")
```

## Report back
The per-source held-out AUC (especially aims_mouse, aims_flu, shehata2019). If the mean beats
~0.57 → the neural net cracked the wall; if not → it didn't, and that's the honest ceiling.
Do NOT tune on the holdout to chase a number.
