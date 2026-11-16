import json,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
V={'abl_nosteer':'--lam 1 --pen 0.5','abl_nobelief':'--alpha 2 --layer 16 --pen 0.5','abl_nopen':'--alpha 2 --layer 16 --lam 1',
   'abl_steer':'--alpha 2 --layer 16','abl_belief':'--lam 1','abl_pen':'--pen 0.5',
   'abl_const':'--alpha 2 --layer 16 --lam 1 --pen 0.5 --bsched const --ssched const'}
SPANS=[(s,s+50) for s in range(0,500,50)]
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def st(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};s.pop('vobj',None);return json.dumps(s,sort_keys=True)
ref=[r['id'] for r in read(O/'sets/chair_test500.jsonl')];assert len(ref)==500
cmd=sys.argv[1]
if cmd=='queue':
    q=[f'chair_test500 chair_abl500 {t} {s} {e} --attn sdpa {a}' for t,a in V.items() for s,e in SPANS]
    (O/'queues/abl_chair.txt').write_text('\n'.join(q)+'\n');print(len(q),'units')
elif cmd=='join':
    full=read(O/'runs/chair_test500_all/s16a2bl1p05-0-500.jsonl');base=read(O/'runs/chair_test500_all/base-0-500.jsonl')
    A=O/'runs/chair_abl500_all';Fd=O/'runs/chair_abl500_vsfull';A.mkdir(exist_ok=True);Fd.mkdir(exist_ok=True)
    def w(d,t,rows):(d/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    w(A,'base',base);w(A,'rbd_full',full);w(Fd,'base',full);w(Fd,'vddbase',base)
    for t in V:
        rows=[]
        for s,e in SPANS:
            f=O/f'runs/chair_abl500/{t}-{s}-{e}.jsonl';r=read(f) if f.exists() else [];assert len(r)==e-s,f'{f} incomplete';rows+=r
        assert [x['id'] for x in rows]==ref,t
        assert len({st(x) for x in rows})==1,f'{t}: rows with different settings'
        w(A,t,rows);w(Fd,t,rows)
    print('joined',['base','rbd_full',*V])
