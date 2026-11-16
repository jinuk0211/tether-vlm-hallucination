import argparse,contextlib,glob,io,json,os,sys
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path);a=ap.parse_args()
BENCH=Path(os.environ.get('BENCH','/home/project-b/LJW/bench'));T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json')
sys.path.insert(0,str(T));import throne_score_aqa as A,pycocotools.coco as coco
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');RUNS=O/'runs';NB=5
ROWS={'Qwen2.5-VL':('qwen',[('Base','base'),('REVIS','revis'),('RUDDER','rudder_clone'),('GrAInS','grains_clone'),('VTI','vti_author'),('TETHER','cf_a3bl2p1')]),
      'Gemma 4 E4B':('gemma',[('Base','base'),('REVIS','revis'),('RUDDER','rudder_clone'),('GrAInS','grains_clone'),('VTI','vti'),('TETHER','cf_a3bl2p1')])}
SETTINGS=((('base','large'),6,'two judges (flan-t5 base, large), threshold 6: main columns'),
          (('base','large','xl'),9,"THRONE's three judges (base, large, xl), threshold 9: xl columns"))
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
full=read(O/'sets/throne_full.jsonl');N=len(full);assert N==1000*NB
keys=[r['metadata']['image_id'] for r in full];rid=[r['id'] for r in full]
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()}
gt={k:set(cid[x['category_id']] for x in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def one(pat):
    fs=glob.glob(pat);assert len(fs)<=1,fs;return fs[0] if fs else None
def jdict(M,tag,J):
    d={}
    for b in range(NB):
        f=one(str(RUNS/f'throne_{M}'/'eval'/tag/f'b{b}-{J}'/'combined'/f'*flan-t5-{J}.json'))
        if f is None:return None
        e=json.load(open(f));assert len(e)==1000 and not set(e)&set(d);d.update(e)
    return d
CAP={}
def captions(M,tag):
    if (M,tag) not in CAP:
        rows=read(RUNS/f'throne_{M}'/f'{tag}-0-{N}.jsonl');assert [r['id'] for r in rows]==rid,(M,tag);CAP[(M,tag)]=rows
    return CAP[(M,tag)]
def tokens(M,tag):return np.array([r['generated_tokens'] for r in captions(M,tag)],dtype=float)
def preds(M,tag,judges):
    dl=[jdict(M,tag,J) for J in judges]
    if any(d is None for d in dl):return None
    by=[{int(k):A.process_aqa_instance_preds(v,num_categories=len(idx),sorted_categories_idx_to_name=idx) for k,v in d.items()} for d in dl]
    return {k:[p[k] for p in by] for k in by[0]}
def mats(pr,thr):
    col,dis=A.majority_vote_collapsing(pr,voting_threshold=thr);P=np.zeros((N,80));C=np.zeros_like(P);G=np.zeros_like(P)
    for i,k in enumerate(keys):
        dq=set(dis[k])
        for g in gt[k]:
            if g not in dq:G[i,n2i[g]]=1
        for p in col[k]:
            if p not in dq:
                P[i,n2i[p]]=1
                if p in gt[k]:C[i,n2i[p]]=1
    return P,C,G
def f(p,r):
    with np.errstate(invalid='ignore',divide='ignore'):return 2*p*r/(p+r),1.25*p*r/(0.25*p+r)
def micro(W,P,C,G):
    np_,nc,ng=P.sum(1),C.sum(1),G.sum(1);ok=(np_>0)&(ng>0);p=np.where(ok,nc/np.maximum(np_,1),0.);r=np.where(ok,nc/np.maximum(ng,1),0.)
    f1,f5=f(p,r);m=~np.isnan(f5);s=W.sum(1)
    return {'P':(W@p)/s,'R':(W@r)/s,'F1':(W[:,m]@f1[m])/W[:,m].sum(1),'F0.5':(W[:,m]@f5[m])/W[:,m].sum(1)}
def cls(W,P,C,G):
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):pr=c_/p_;rc=c_/g_;f1,f5=f(pr,rc);return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F1':np.nanmean(f1,1),'F0.5':np.nanmean(f5,1)}
Q=[('ALL','P'),('ALL','R'),('ALL','F1'),('ALL','F0.5'),('CLS','P'),('CLS','R'),('CLS','F1'),('CLS','F0.5')]
def metr(W,m):
    with np.errstate(invalid='ignore'):x,y=micro(W,*m),cls(W,*m)
    return [100*(x if s=='ALL' else y)[q] for s,q in Q]
rng=np.random.default_rng(0);W=np.stack([np.bincount(rng.integers(0,N,N),minlength=N) for _ in range(2000)]).astype(float);one_=np.ones((1,N))
out={'N':N,'results':{}}
for judges,thr,label in SETTINGS:
    print(f'##### {N} images; {label}')
    for bb,(M,rows) in ROWS.items():
        print(f'== {bb}');print('   '+' '*22+''.join(f'{s+"_"+q:>13s}' for s,q in Q)+'     Tok')
        ms={}
        for name,tag in rows:
            pr=preds(M,tag,judges);ms[name]=None if pr is None else mats(pr,thr)
        B0=metr(W,ms['Base']) if ms['Base'] is not None else None
        for name,tag in rows:
            m=ms[name]
            if m is None:print(f'   {name:22s} missing judgments');continue
            v=[float(x[0]) for x in metr(one_,m)];tk=float(tokens(M,tag).mean())
            print(f'   {name:22s}'+''.join(f'{x:13.1f}' for x in v)+f'{tk:8.0f}')
            rec={'metrics':dict(zip([s+'_'+q for s,q in Q],v)),'tokens':tk}
            if B0 is not None and name!='Base':
                ci=[(float(np.quantile(x-y,.025)),float(np.quantile(x-y,.975))) for x,y in zip(metr(W,m),B0)]
                print(f"   {'':22s}"+''.join(f"{'[%+.1f,%+.1f]'%c:>13s}" for c in ci));rec['ci_vs_base']=dict(zip([s+'_'+q for s,q in Q],ci))
            out['results'].setdefault(f'{len(judges)}j_thr{thr}',{}).setdefault(bb,{})[name]=rec
    print()
for bb,(M,rows) in ROWS.items():
    b,r=captions(M,'base'),captions(M,'revis')
    print(f'{bb}: REVIS caption == Base caption on {sum(x["prediction"]==y["prediction"] for x,y in zip(b,r))}/{N} rows')
if a.json:a.json.write_text(json.dumps(out,indent=1))
