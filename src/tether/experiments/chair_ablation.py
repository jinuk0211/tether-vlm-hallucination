import json,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
V={'abl_nosteer':'--lam 1 --pen 0.5','abl_nobelief':'--alpha 2 --layer 16 --pen 0.5','abl_nopen':'--alpha 2 --layer 16 --lam 1',
   'abl_steer':'--alpha 2 --layer 16','abl_belief':'--lam 1','abl_pen':'--pen 0.5',
   'abl_const':'--alpha 2 --layer 16 --lam 1 --pen 0.5 --bsched const --ssched const'}
REUSE={'abl_nopen':(O/'runs/chair_test60_sweep/s16a2bl1-0-60.jsonl',60),'abl_belief':(O/'runs/chair_test150_all/bl1-0-150.jsonl',150)}
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def st(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};s.pop('vobj',None);return json.dumps(s,sort_keys=True)
ref=[r['id'] for r in read(O/'sets/chair_test500.jsonl')];assert len(ref)==500
def spans(t):
    a=REUSE[t][1] if t in REUSE else 0;b=[a]+list(range((a//50+1)*50,500,50)) if a%50 else list(range(a,500,50));return [(s,min(s+50,500) if s%50==0 else (s//50+1)*50) for s in b]
cmd=sys.argv[1]
if cmd=='queue':
    q=['# Qwen CHAIR ablation of RBD s16a2bl1p05 (experiments/chair_ablation.py)']
    for t,a in V.items():
        for s,e in spans(t):q.append(f'chair_test500 chair_abl500 {t} {s} {e} --attn sdpa {a}')
    (O/'queues/abl_chair.txt').write_text('\n'.join(q)+'\n');print(len(q)-1,'units',{t:sum(e-s for s,e in spans(t)) for t in V})
elif cmd=='join':
    full=read(O/'runs/chair_test500_all/s16a2bl1p05-0-500.jsonl');base=read(O/'runs/chair_test500_all/base-0-500.jsonl')
    A=O/'runs/chair_abl500_all';Fd=O/'runs/chair_abl500_vsfull';A.mkdir(exist_ok=True);Fd.mkdir(exist_ok=True)
    def w(d,t,rows):(d/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    w(A,'base',base);w(A,'rbd_full',full);w(Fd,'base',full);w(Fd,'vddbase',base)
    for t in V:
        rows=[]
        if t in REUSE:rows=read(REUSE[t][0])
        for s,e in spans(t):
            f=O/f'runs/chair_abl500/{t}-{s}-{e}.jsonl';r=read(f) if f.exists() else [];assert len(r)==e-s,f'{f} incomplete';rows+=r
        assert [x['id'] for x in rows]==ref,t
        assert len({st(x) for x in rows})==1,f'{t}: settings differ between reused and new rows'
        w(A,t,rows);w(Fd,t,rows)
    print('joined',['base','rbd_full',*V])
