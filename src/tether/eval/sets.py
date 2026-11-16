import glob,hashlib,json,os,sys,tempfile
from pathlib import Path
import numpy as np

R=Path('/home/project-b/LJW/vlm-grounding-steering')
STATE={'chair_dev100':R/'data/chair-dev-train2014-100/chair-scorer-state.json',
       'chair_test500':Path(os.environ.get('RBD2_CHAIR_TEST_STATE',R/'data/hallucination-suite/chair-scorer-state.json'))}
SETS=Path(os.environ.get('RBD2_SETS',R/'outputs/rbd2-20261102/sets'))
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
rng=np.random.default_rng(0)

def chair(refs,runs):
    from tether.eval.common import _load_external_module
    src=R/'upstream/chair/chair.py';c=json.loads(STATE[SET].read_text())
    assert c['built_with_unmodified_constructor'] and c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
    mod=_load_external_module(src,'chair_sets');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
    sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
    ids=[r['metadata']['image_id'] for r in refs];res={};per={}
    for name,rows in runs.items():
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:
            json.dump([{'image_id':r['metadata']['image_id'],'caption':rows[r['id']]['prediction']} for r in refs],f)
        d=sc.compute_chair(f.name,'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];Rc=m['Recall']
        s={x['image_id']:x for x in d['sentences']}
        per[name]={'hal':np.array([len(s[i]['mscoco_hallucinated_words'])>0 for i in ids],float),
                   'n_gt':np.array([len(set(s[i]['mscoco_gt_words'])) for i in ids],float),
                   'n_hit':np.array([len(set(s[i]['mscoco_gt_words'])&set(s[i]['mscoco_generated_words'])) for i in ids],float)}
        res[name]={'CHAIR_s':100*m['CHAIRs'],'CHAIR_i':100*m['CHAIRi'],'recall':100*Rc,'F1':100*2*P*Rc/(P+Rc)}
    B=[rng.integers(0,len(ids),len(ids)) for _ in range(2000)];b=per['base']
    for n,x in per.items():
        if n=='base':continue
        ds=[100*(x['hal'][k].mean()-b['hal'][k].mean()) for k in B]
        dr=[100*(x['n_hit'][k].sum()/x['n_gt'][k].sum()-b['n_hit'][k].sum()/b['n_gt'][k].sum()) for k in B]
        res[n]['ci']={'CHAIR_s':[float(np.quantile(ds,q)) for q in (.025,.975)],'recall':[float(np.quantile(dr,q)) for q in (.025,.975)]}
    return res

def main(rundir):
    refs=read(SETS/f'{SET}.jsonl');ids=[r['id'] for r in refs];N=len(ids);runs={};info={}
    for f in sorted(glob.glob(str(Path(rundir)/f'*-0-{N}.jsonl'))):
        tag=Path(f).name[:-len(f'-0-{N}.jsonl')];rows={r['id']:r for r in read(f)}
        if not all(i in rows for i in ids):continue
        runs[tag]=rows;r0=rows[ids[0]]
        info[tag]={'setting':r0.get('setting'),'tokens':float(np.mean([rows[i]['generated_tokens'] for i in ids])),
                   'at512':int(sum(rows[i]['generated_tokens']>=512 for i in ids))}
    assert 'base' in runs,'no complete base run'
    runs={'base':runs.pop('base'),**runs}
    res=chair(refs,runs)
    for t in res:res[t].update(info[t])
    Path(rundir,'score.json').write_text(json.dumps({'set':SET,'n':len(ids),'results':res},indent=1))
    keys=('CHAIR_s','CHAIR_i','recall','F1')
    for t,v in res.items():
        ci=' '.join(f"d{q}[{c[0]:+.1f},{c[1]:+.1f}]" for q,c in v.get('ci',{}).items())
        print(f"{t:28s} "+' '.join(f"{k} {v[k]:5.1f}" for k in keys)+f"  tok {v['tokens']:4.0f} @512 {v['at512']:2d}  {ci}")

if __name__=='__main__':SET=sys.argv[1];main(sys.argv[2])
