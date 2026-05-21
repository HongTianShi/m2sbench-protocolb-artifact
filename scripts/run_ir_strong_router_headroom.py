import sys, json, warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_standard_ir_access_audit import (load_collection, encode_or_load, search_centroid, search_ivfpq, search_flat, per_query_ndcg, score_features, safe_id)
DATASETS=['beir/fiqa/test','beir/scifact/test','beir/nfcorpus/test','beir/arguana','antique/test']
model='sentence-transformers/all-MiniLM-L6-v2'; lambda_cost=0.08; costs=np.array([0.0,0.20,0.58], dtype='float32'); views=np.array(['summary','pq','full'])
Xs=[]; Ys=[]; Ps=[]; train_masks=[]; dataset_names=[]
for di,dataset in enumerate(DATASETS):
    print('load', dataset, flush=True)
    report=json.loads((ROOT/'reports'/f'standard_ir_access_{safe_id(dataset)}.json').read_text(encoding='utf-8'))
    docs, doc_ids, queries, qids, rels = load_collection(dataset,0,0,13)
    doc_emb, query_emb = encode_or_load(dataset, model, docs, queries, 256)
    nlist=int(report['nlist']); pq_m=int(report['pq_m']); pq_nbits=int(report['pq_nbits']); nprobe=int(report['nprobe'])
    scores_s, ids_s, _ = search_centroid(doc_emb, query_emb, 10, nlist)
    scores_p, ids_p, _ = search_ivfpq(doc_emb, query_emb, 10, nlist, pq_m, pq_nbits, nprobe)
    scores_f, ids_f, _ = search_flat(doc_emb, query_emb, 10)
    per=np.column_stack([per_query_ndcg(ids_s,rels,10), per_query_ndcg(ids_p,rels,10), per_query_ndcg(ids_f,rels,10)]).astype('float32')
    util=per - lambda_cost*costs[None,:]
    f_s=score_features(scores_s); f_p=score_features(scores_p)
    onehot=np.zeros((len(queries), len(DATASETS)), dtype='float32'); onehot[:,di]=1
    q=query_emb.astype('float32')
    qhead=q[:,:min(128,q.shape[1])]
    qstats=np.column_stack([q.mean(1), q.std(1), q.min(1), q.max(1), np.mean(q>0,1)]).astype('float32')
    X=np.column_stack([onehot,qhead,qstats,f_s,f_p]).astype('float32')
    n=len(queries); cut=max(1,min(n-1,int(round(n*.6))))
    train=np.zeros(n,dtype=bool); train[:cut]=True
    Xs.append(X); Ys.append(util); Ps.append(per); train_masks.append(train); dataset_names.extend([dataset]*n)
X=np.vstack(Xs); Y=np.vstack(Ys); P=np.vstack(Ps); train=np.concatenate(train_masks); test=~train
print('total', X.shape, 'train', int(train.sum()), 'test', int(test.sum()), flush=True)
labels=Y.argmax(1)
def paired_ci(diff, reps=2000, seed=7):
    diff=np.asarray(diff,dtype='float32')
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,dtype='float32')
    n=len(diff)
    for b in range(reps):
        vals[b]=diff[rng.integers(0,n,n)].mean()
    return {"mean":float(diff.mean()),"lo":float(np.quantile(vals,.025)),"hi":float(np.quantile(vals,.975))}

def eval_choices(name, choices):
    idx=np.array(choices, dtype=int)
    rows=np.arange(len(idx))
    util=Y[test][rows, idx]
    ndcg=P[test][rows, idx]
    cost=costs[idx]
    oracle=Y[test].max(1)
    return {'name':name,'queries':int(test.sum()),'utility':float(util.mean()),'ndcg':float(ndcg.mean()),'cost':float(cost.mean()),'regret':float((oracle-util).mean()),'choices':{str(views[i]):float(np.mean(idx==i)) for i in range(3)},'_utility_vector':util.astype(float).tolist()}
results=[]
for i,v in enumerate(views): results.append(eval_choices('fixed_'+str(v), np.full(test.sum(), i)))
oracle_idx=Y[test].argmax(1)
oracle_util=Y[test].max(1)
results.append({'name':'oracle_route','queries':int(test.sum()),'utility':float(oracle_util.mean()),'ndcg':float(P[test][np.arange(test.sum()),oracle_idx].mean()),'cost':float(costs[oracle_idx].mean()),'regret':0.0,'choices':{str(views[i]):float(np.mean(oracle_idx==i)) for i in range(3)},'_utility_vector':oracle_util.astype(float).tolist()})
reg_models=[
 ('ridge_reg', make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))),
 ('extra_trees_reg', ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, max_features='sqrt', random_state=7, n_jobs=-1)),
 ('rf_reg', RandomForestRegressor(n_estimators=600, min_samples_leaf=2, max_features='sqrt', random_state=9, n_jobs=-1)),
 ('hist_gbr_reg', MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, l2_regularization=0.1, min_samples_leaf=10, random_state=10))),
 ('mlp_reg', make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256,128), alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=11)))
]
clf_models=[
 ('extra_trees_cls', ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=21, n_jobs=-1)),
 ('rf_cls', RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=23, n_jobs=-1)),
 ('hist_gbc_cls', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, l2_regularization=0.1, min_samples_leaf=10, random_state=24)),
 ('logistic_cls', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))),
 ('mlp_cls', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,128), alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=25)))
]
for name,m in reg_models:
    print('fit', name, flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X[train], Y[train])
    pred=m.predict(X[test])
    if pred.ndim==1: pred=pred.reshape(-1,3)
    results.append(eval_choices(name, pred.argmax(1)))
for name,m in clf_models:
    print('fit', name, flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X[train], labels[train])
    pred=m.predict(X[test])
    results.append(eval_choices(name, pred))
full_vec=np.array(next(r for r in results if r['name']=='fixed_full')['_utility_vector'], dtype='float32')
for r in results:
    if r['name']!='fixed_full':
        r['paired_diff_vs_fixed_full']=paired_ci(np.array(r['_utility_vector'], dtype='float32')-full_vec)
public_results=[]
results=sorted(results, key=lambda r:(-r['utility'], r['cost']))
for r in results:
    print(r['name'], 'util', round(r['utility'],4), 'ndcg', round(r['ndcg'],4), 'cost', round(r['cost'],4), 'regret', round(r['regret'],4), r['choices'], flush=True)
    public_results.append({k:v for k,v in r.items() if k!='_utility_vector'})
out={'task':'ir_strong_router_headroom','lambda':lambda_cost,'features':'dataset_onehot+query_emb128+query_stats+summary_scores+pq_scores','results':public_results}
(ROOT/'reports'/'ir_strong_router_headroom.json').write_text(json.dumps(out,indent=2), encoding='utf-8')
lines=['# IR Strong Router and Oracle Headroom','','Held-out pooled 60/40 split over the five public standard-qrel datasets. Features are method-visible: dataset ID, query embedding summaries, summary score summaries, and declared PQ score summaries for B1-style rows. Utility is NDCG@10 - 0.08 C.','', '| policy | utility | NDCG | cost | regret | choices | paired diff vs fixed full |','|---|---:|---:|---:|---:|---|---:|']
for r in public_results:
    ci=r.get('paired_diff_vs_fixed_full')
    diff='--' if ci is None else f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
    ch=', '.join(f"{k} {v:.2f}" for k,v in r['choices'].items())
    lines.append(f"| {r['name']} | {r['utility']:.4f} | {r['ndcg']:.4f} | {r['cost']:.4f} | {r['regret']:.4f} | {ch} | {diff} |")
(ROOT/'reports'/'ir_strong_router_headroom.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
