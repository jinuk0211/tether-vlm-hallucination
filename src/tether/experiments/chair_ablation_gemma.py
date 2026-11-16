import json,os,sys
from pathlib import Path
O=Path(os.environ.get('GABL_L','/home/project-b/LJW'))/'vlm-grounding-steering/outputs/rbd2-20261102'
FULL='s24a1bl1p05';S='--alpha 1 --layer 24'
V={'abl_nosteer':'--lam 1 --pen 0.5','abl_nobelief':f'{S} --pen 0.5','abl_nopen':f'{S} --lam 1',
   'abl_steer':S,'abl_belief':'--lam 1','abl_pen':'--pen 0.5',
   'abl_const':f'{S} --lam 1 --pen 0.5 --bsched const --ssched const'}
SPANS=[(s,s+50) for s in range(0,500,50)]
FF=O/f'runs/gchair_test500_all/{FULL}-0-500.jsonl';BF=O/'runs/gchair_test500_all/base-0-500.jsonl';TF=O/'runs/gchair_test500_tbl/base-0-500.jsonl'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def st(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};s.pop('vobj',None);return json.dumps(s,sort_keys=True)
def want(**k):return json.dumps({'lam':0.0,'clip':5.0,'pen':0.0,'bsched':'decay','alpha':0.0,'layer':16,'ssched':'decay','r':'measured','attn':'eager','bpos':False,'norm':'raw',**k},sort_keys=True)
ref=[r['id'] for r in read(O/'sets/chair_test500.jsonl')];assert len(ref)==500
cmd=sys.argv[1]
if cmd=='check':
    f=read(FF);assert sorted(x['id'] for x in f)==sorted(ref),'full row incomplete';assert {st(x) for x in f}=={want(alpha=1.0,layer=24,lam=1.0,pen=0.5)},f'full row setting: {st(f[0])}'
    b=read(BF);assert sorted(x['id'] for x in b)==sorted(ref),'vdd Base incomplete';s=json.loads(st(b[0]));assert s['alpha']==s['lam']==s['pen']==0 and s['attn']=='eager',f'vdd Base setting: {s}'
    t=read(TF);assert sorted(x['id'] for x in t)==sorted(ref),'table Base incomplete'
    print('check ok: full, vdd Base, table Base complete')
elif cmd=='queue':
    q=[f'chair_test500 gabl_chair500 {t} {s} {e} --attn eager {a}' for t,a in V.items() for s,e in SPANS]
    (O/'queues/gabl_chair.txt').write_text('\n'.join(q)+'\n');print(len(q),'units')
elif cmd=='join':
    full=read(FF);base=read(BF);tbl=read(TF)
    A=O/'runs/gabl_chair500_all';Fd=O/'runs/gabl_chair500_vsfull';A.mkdir(exist_ok=True);Fd.mkdir(exist_ok=True)
    def w(d,t,rows):(d/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    w(A,'base',base);w(A,'rbd_full',full);w(Fd,'base',full);w(Fd,'vddbase',base);w(Fd,'tblbase',tbl)
    for t in V:
        rows=[]
        for s,e in SPANS:
            f=O/f'runs/gabl_chair500/{t}-{s}-{e}.jsonl';r=read(f) if f.exists() else [];assert len(r)==e-s,f'{f} incomplete';rows+=r
        assert [x['id'] for x in rows]==ref,t
        assert len({st(x) for x in rows})==1,f'{t}: rows with different settings'
        w(A,t,rows);w(Fd,t,rows)
    print('joined',['base','rbd_full',*V])
elif cmd=='table':
    X=json.loads((Path(sys.argv[2])/'score.json').read_text())['results']
    def nums(t):v=X[t];return f"{v['CHAIR_s']:.1f} & {v['CHAIR_i']:.1f} & {v['recall']:.1f}"
    def row(n,t):c=X[t]['ci'];f=lambda q:f"[${c[q][0]:+.1f}$, ${c[q][1]:+.1f}$]";return f"    {n} & {nums(t)} & {f('CHAIR_s')} & {f('recall')} \\\\"
    print('\n'.join([r'\begin{table*}[t]',r'  \centering',r'  \caption{Ablation Study on Proposed RBD Components (Gemma 4 E4B)}',r'  \label{tab:ablation_study_gemma}',r'  \small',
      r'  \setlength{\tabcolsep}{4pt}',r'  \begin{tabular}{lccccc}',r'    \toprule',
      r'    \textbf{Configuration} & \textbf{CHAIR$_s$ ($\downarrow$)} & \textbf{CHAIR$_i$ ($\downarrow$)} & \textbf{Recall ($\uparrow$)} & \textbf{$\Delta$CHAIR$_s$ vs. Full} & \textbf{$\Delta$Recall vs. Full} \\',
      r'    \midrule',f"    Full (RBD) & {nums('base')} & -- & -- \\\\",r'    \midrule',
      row('$-$ steering','abl_nosteer'),row('$-$ penalty','abl_nopen'),row('$-$ belief','abl_nobelief'),r'    \midrule',
      row('w/o decay schedule','abl_const'),row('belief only','abl_belief'),row('penalty only','abl_pen'),row('steering only','abl_steer'),r'    \midrule',
      row('Base (Vanilla)','vddbase'),r'    \bottomrule',r'  \end{tabular}',r'\end{table*}']))
