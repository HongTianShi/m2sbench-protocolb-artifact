import numpy as np

def retrieval_metrics(query_labels, ranked_labels, ks=(1,5)):
    out={}
    n=len(query_labels)
    for k in ks:
        hit=0
        for q, ranks in zip(query_labels, ranked_labels):
            hit += int(q in ranks[:k])
        out[f"recall_at_{k}"]=float(hit/n) if n else 0.0
    aps=[]
    for q, ranks in zip(query_labels, ranked_labels):
        rel=[1 if r==q else 0 for r in ranks]
        hits=0; prec=[]
        for i,v in enumerate(rel, start=1):
            if v:
                hits += 1; prec.append(hits / i)
        aps.append(sum(prec)/max(sum(rel),1))
    out["map"]=float(np.mean(aps)) if aps else 0.0
    return out
