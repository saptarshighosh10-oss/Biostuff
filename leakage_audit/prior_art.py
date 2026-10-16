import json, time, urllib.parse, subprocess
CA="/opt/anaconda3/lib/python3.13/site-packages/certifi/cacert.pem"; M="saptarshi.ghosh10@gmail.com"
def get(path, **p):
    p.update(mailto=M); u="https://api.openalex.org/"+path+"?"+urllib.parse.urlencode(p)
    for _ in range(6):
        d=json.loads(subprocess.run(["curl","-s","--cacert",CA,u],capture_output=True,text=True).stdout or "{}")
        if "error" not in d: return d
        time.sleep(40)
    return {"results":[]}
def show(rs): 
    for r in rs: print(" ",r.get("publication_year"),r.get("cited_by_count"),(r.get("title") or "")[:105],"|",r.get("doi"))
S="title,publication_year,cited_by_count,doi,id"
for q in ["antibody developability prediction data leakage","leakage susceptibility benchmark overlap","EuroSAT duplicate leakage","homology leakage protein property prediction split","Ginkgo antibody developability competition GDPa1 cross-validation","train test overlap inflation predicted from synthetic contamination"]:
    print("==",q); show(get("works",search=q,**{"per-page":8,"select":S})["results"]); time.sleep(2)
for name,q in [("Rosenblatt 2024","Data leakage inflates prediction performance in connectome-based machine learning models"),("FLAb 2024","FLAb: Benchmarking deep learning methods for antibody fitness prediction"),("Elangovan 2021","Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation")]:
    w=get("works",search=q,**{"per-page":1,"select":"id,title"})["results"]
    if not w: continue
    wid=w[0]["id"].split("/")[-1]; print("== cited_by of",name,"(",w[0]["title"][:60],")")
    d=get("works",filter=f"cites:{wid},from_publication_date:2024-01-01",search="leakage OR overlap OR contamination OR redundancy OR susceptibility OR inflation",**{"per-page":12,"select":S,"sort":"cited_by_count:desc"})
    show(d["results"]); time.sleep(2)
