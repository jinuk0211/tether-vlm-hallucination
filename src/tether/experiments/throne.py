import glob,json,sys
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';S=O/'sets';Q=O/'queues';RUNS=O/'runs'
V=str(R/'outputs/rbd-20261030/collect')
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
METH={'qwen':[('qb','vti_author',50,'--attn eager'),
              ('v4q','cf_a3bl2p1',100,f'--attn sdpa --alpha 3 --layer 16 --lam 2 --pen 1 --vobj {V}/qwen-vobj2cf.json'),
              ('qb','grains_clone',200,'--attn eager'),('qb','rudder_clone',200,'--attn sdpa'),('qb','base',200,'--attn sdpa'),('qb','revis',200,'--attn sdpa')],
      'gemma':[('v4g','cf_a3bl2p1',50,f'--attn eager --alpha 3 --layer 24 --lam 2 --pen 1 --vobj {V}/gemma-vobj2cf.json'),
               ('gb','rudder_clone',100,''),('gb','base',100,''),('gb','revis',100,''),('gb','vti',100,''),('gb','grains_clone',100,'')]}
BLOCK=1000;NB=5
def parts(M,tag):
    rows={}
    for f in glob.glob(str(RUNS/f'throne_parts_{M}'/f'{tag}-*-*.jsonl')):
        if Path(f).name[:-6].rsplit('-',2)[0]!=tag:continue
        for r in read(f):rows[r['id']]=r
    return rows
cmd=sys.argv[1]
if cmd=='queues':
    dec=[];jud=[]
    for b in range(NB):
        for i in range(max(len(METH['qwen']),len(METH['gemma']))):
            for M in ('gemma','qwen'):
                if i>=len(METH[M]):continue
                k,t,c,a=METH[M][i]
                for s in range(b*BLOCK,(b+1)*BLOCK,c):dec.append(f'{k} throne_full throne_parts_{M} {t} {s} {s+c} {a}'.rstrip())
        for J in ('base','large','xl'):
            for M in ('qwen','gemma'):
                for k,t,c,a in METH[M]:jud.append(f'{M} {t} {b} {J} '+','.join(f'{s}-{s+c}' for s in range(b*BLOCK,(b+1)*BLOCK,c)))
    (Q/'throne_decode.txt').write_text('\n'.join(dec)+'\n');(Q/'throne_judge.txt').write_text('\n'.join(jud)+'\n')
    print('decode units',len(dec),'judge units',len(jud))
elif cmd=='join':
    refs=read(S/'throne_full.jsonl');N=len(refs);assert N==NB*BLOCK;ok=True
    for M in ('qwen','gemma'):
        (RUNS/f'throne_{M}').mkdir(exist_ok=True)
        for k,t,c,a in METH[M]:
            rows=parts(M,t);miss=[r['id'] for r in refs if r['id'] not in rows]
            if miss:print(M,t,'missing',len(miss));ok=False;continue
            out=[rows[r['id']] for r in refs];assert all(o['metadata']['image_id']==r['metadata']['image_id'] for o,r in zip(out,refs))
            (RUNS/f'throne_{M}'/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in out));print(M,t,'joined',N)
    sys.exit(0 if ok else 1)
elif cmd=='status':
    print(f'{"model":6s} {"tag":13s} {"rows":>5s}/{NB*BLOCK}  {"s/row":>6s}  judged blocks base/large/xl')
    for M in ('qwen','gemma'):
        for k,t,c,a in METH[M]:
            rows=parts(M,t);el=[r['elapsed_seconds'] for r in rows.values()]
            jb=[sum(1 for b in range(NB) if glob.glob(str(RUNS/f'throne_{M}'/'eval'/t/f'b{b}-{J}'/'combined'/f'*flan-t5-{J}.json'))) for J in ('base','large','xl')]
            print(f'{M:6s} {t:13s} {len(rows):5d}/{NB*BLOCK}  {(sum(el)/len(el) if el else 0):6.1f}  {jb[0]}/{jb[1]}/{jb[2]}')
