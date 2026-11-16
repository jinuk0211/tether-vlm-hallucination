import glob,hashlib,json,os,re
from pathlib import Path
L=Path(os.environ.get('HAT_L','/home/project-b/LJW'));R=L/'vlm-grounding-steering';O=R/'outputs/rbd2-20261102'
RUN=os.environ.get('HAT_RUN','hat385');H=O/RUN;SET=O/f'sets/{RUN}.jsonl';HAT_JSON=L/'bench/aloha/data/hat-trainval.json'
def rundir(model):return O/'runs'/f'{RUN}_{model}'
def read(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def key(kind,text):return hashlib.sha256(f'{kind}\0{text}'.encode()).hexdigest()
def parser_file(parser):return O/'hat_parse'/(re.sub(r'[^\w.-]+','_',parser)+f'-{RUN}.jsonl')
def load_cache(parser):
    f=parser_file(parser);return {r['key']:r['raw'] for r in read(f)} if f.exists() else {}

def load_runs(run_dir,ids):
    by={}
    for f in glob.glob(str(Path(run_dir)/'*.jsonl')):
        m=re.fullmatch(r'(.+)-(\d+)-(\d+)\.jsonl',Path(f).name)
        if m:by.setdefault(m[1],{}).update({r['id']:r for r in read(f)})
    done={t:{i:x[i] for i in ids} for t,x in by.items() if all(i in x for i in ids)}
    for t,x in done.items():
        assert len({json.dumps({'bpos':False,'norm':'raw',**(r.get('setting') or {})},sort_keys=True) for r in x.values()})==1,f'{t}: mixed settings'
    return done,{t:sum(i in x for i in ids) for t,x in by.items() if t not in done}
