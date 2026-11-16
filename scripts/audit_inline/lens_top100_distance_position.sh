#!/bin/bash
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= .venv/bin/python - <<'P'
import torch,numpy as np
O='outputs/logit-lens-20261104';D=torch.load(f'{O}/qwen-topk-0-500.pt',weights_only=False);nodes=D['nodes'];ni={n:i for i,n in enumerate(nodes)};recs=D['records'];band=list(range(22,28));N=len(nodes)
DB=[1,5,17,65,10**6];DL=['1-4','5-16','17-64','65+'];PBk=[0,64,128,10**6];PL=['0-63','64-127','128+']
K=(1,10,100)
absH={k:np.zeros((3,N)) for k in K};absP=np.zeros((3,N))
rows={c:{k:np.zeros((4,3)) for k in K} for c in ('hall','correct')};pos={c:np.zeros((4,3)) for c in ('hall','correct')};exc={c:{k:np.zeros((4,3)) for k in K} for c in ('hall','correct')}
obj={c:set() for c in rows};per=[]
parsed=[]
for r in recs:
    ca=np.array(r['caption_ids']);prev=np.r_[-1,ca[:-1]];h=r['hits'];h=h[np.isin(h[:,0],band)];h=h[(h[:,4]!=ca[h[:,1]])&(h[:,4]!=prev[h[:,1]])]
    S={k:set(map(tuple,h[h[:,2]<k][:,[1,3]].tolist())) for k in K};parsed.append((r,S))
    L=r['labels'];GT,M=set(L['gt']),set(L['mentioned'])
    for n,i in ni.items():
        if n in GT or n in M:continue
        for t in range(r['T']):
            b=np.digitize(t,PBk)-1;absP[b,i]+=1
            for k in K:absH[k][b,i]+=(t,i) in S[k]
base={k:absH[k]/np.maximum(absP,1) for k in K}
for r,S in parsed:
    H=set(r['labels']['hallucinated']);seen={}
    for x in r['mentions']:
        if x['first'] and x['node'] in ni:seen[x['node']]=(x['t'],'hall' if x['node'] in H else 'correct')
    for n,(t0,c) in seen.items():
        i=ni[n];obj[c].add((r['image_id'],n))
        for t in range(t0):
            d=np.digitize(t0-t,DB)-1;b=np.digitize(t,PBk)-1;pos[c][d,b]+=1
            for k in K:
                y=(t,i) in S[k];rows[c][k][d,b]+=y;exc[c][k][d,b]+=y-base[k][b,i]
for c in ('hall','correct'):
    print(f'== {c}: {len(obj[c])} objects; % of positions before writing, raw / chance-removed; rows = tokens until it is written, cols = caption position {PL}')
    print('  positions',{DL[d]:[int(x) for x in pos[c][d]] for d in range(4)})
    for k in K:
        print(f'  top-{k}')
        for d in range(4):
            print(f'   {DL[d]:6s} '+' | '.join(f'{100*rows[c][k][d,b]/pos[c][d,b]:5.2f}/{100*exc[c][k][d,b]/pos[c][d,b]:5.2f}' if pos[c][d,b]>=30 else '   n/a     ' for b in range(3)))
P
