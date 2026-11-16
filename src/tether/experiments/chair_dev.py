import json,os,sys
from pathlib import Path
L=Path('/home/project-b/LJW');R=L/'vlm-grounding-steering';O=R/'outputs/rbd2-20261102';RUN='chair_dev100';N=100
SET=O/f'sets/{RUN}.jsonl'
METHODS={'qwen':[('bench','base','--attn sdpa'),('vdd','vddbase','--attn sdpa'),('vdd','s16a2bl1p05','--attn sdpa --layer 16 --alpha 2 --lam 1 --pen 0.5'),
                 ('bench','revis','--attn sdpa'),('bench','rudder_clone','--attn sdpa'),('bench','grains_clone','--attn eager'),('vti','vtiws_v0.9_t0.9','')],
         'qwen3':[('bench','base','--attn sdpa'),('vdd','vddbase','--attn sdpa'),('vdd','s20a1bl1p05','--attn sdpa --layer 20 --alpha 1 --lam 1 --pen 0.5'),
                  ('grains','revis_official','--attn sdpa'),('bench','rudder_clone','--attn sdpa'),('grains','grains_clone','--attn eager'),('bench','vti','--attn sdpa')],
         'gemma':[('bench','base',''),('vdd','vddbase','--attn eager'),('vdd','s24a1bl1p05','--attn eager --layer 24 --alpha 1 --lam 1 --pen 0.5'),
                  ('bench','revis',''),('bench','rudder_clone',''),('bench','grains_clone',''),('bench','vti','')]}
def read(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def rundir(m):return O/'runs'/f'{RUN}_{m}'

def prep():
    rows=read(R/'data/chair-dev-train2014-100/chair.jsonl');assert len(rows)==N
    SET.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    q=[f'{m} {k} {t} 0 {N} {a}'.rstrip() for m,ms in METHODS.items() for k,t,a in ms]
    p=O/f'queues/{RUN}.txt';p.write_text('\n'.join(q)+'\n',newline='\n');print(p.name,len(q),'units')

def run_scorer(d):
    os.environ.setdefault('RBD2_SETS',str(O/'sets'))
    from tether.eval import sets as S
    S.SET=RUN;S.main(d)

def score(m):
    ids=[r['id'] for r in read(SET)];rows={}
    for k,t,a in METHODS[m]:
        f=rundir(m)/f'{t}-0-{N}.jsonl';by={x['id']:x for x in read(f)} if f.exists() else {}
        if all(i in by for i in ids):rows[t]=[by[i] for i in ids]
        else:print(m,t,'INCOMPLETE',sum(i in by for i in ids),'rows')
    assert 'base' in rows and 'vddbase' in rows,f'{m}: no complete Bases: {sorted(rows)}'
    A,B=Path(f'{rundir(m)}_all'),Path(f'{rundir(m)}_vdd');A.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
    def w(d,t,r):(d/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in r),encoding='utf-8')
    for t,r in rows.items():
        w(A,t,r)
        if t=='vddbase':w(B,'base',r)
        elif t!='base':w(B,t,r)
    print(f'=== {m}: paired bootstrap vs bench Base');run_scorer(A)
    print(f'=== {m}: paired bootstrap vs vdd Base');run_scorer(B)
    a=json.loads((A/'score.json').read_text())['results'];b=json.loads((B/'score.json').read_text())['results']
    names={'base':'Base','revis':'REVIS','revis_official':'REVIS','rudder_clone':'RUDDER','grains_clone':'GrAInS','vti':'VTI','vtiws_v0.9_t0.9':'VTI'}
    def cell(v,c,q,f):
        x=c.get('ci',{}).get(q);return f'{v[q]:{f}}'+('$^{*}$' if x and (x[0]>0 or x[1]<0) else '')
    out=[]
    for k,t,_ in METHODS[m]:
        if t not in a:continue
        vdd=k in ('vdd','vti');v=a[t];c=b.get(t,{}) if vdd else v
        out.append(f"{'Base, RBD code path' if t=='vddbase' else 'RBD (Ours)' if k=='vdd' else names[t]} & {cell(v,c,'CHAIR_s','.1f')} & {v['CHAIR_i']:.2f} & {cell(v,c,'recall','.1f')} & {v['tokens']:.0f} \\\\")
    (O/f'logs/cdev_table_{m}.tex').write_text('\n'.join(out)+'\n',encoding='utf-8');print('\n'.join(out))

if __name__=='__main__':
    c=sys.argv[1]
    if c=='prep':prep()
    else:score(sys.argv[2])
