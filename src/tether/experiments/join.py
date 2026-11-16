import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
SET,SRC,OUT=sys.argv[1],O/'runs'/sys.argv[2],O/'runs'/sys.argv[3];OUT.mkdir(parents=True,exist_ok=True)
ids=[json.loads(l)['id'] for l in open(O/f'sets/{SET}.jsonl') if l.strip()];N=len(ids);ok=True
for t in sys.argv[4:]:
    by={}
    for f in glob.glob(str(SRC/f'{t}-*-*.jsonl')):
        if re.fullmatch(re.escape(t)+r'-\d+-\d+\.jsonl',Path(f).name):by.update({r['id']:r for r in map(json.loads,open(f))})
    n=sum(i in by for i in ids)
    if n==N:
        assert len({json.dumps(by[i].get('setting'),sort_keys=True) for i in ids})==1,f'{t}: rows with different settings'
        (OUT/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
    else:ok=False
    print('join',t,n,'/',N,'complete' if n==N else 'INCOMPLETE')
sys.exit(0 if ok else 1)
