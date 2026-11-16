import json,shutil,sys
from pathlib import Path
from tether.eval import throne as S
S.JUDGES=[str(S.BENCH/'models/flan-t5-xl')]
gpu,rundir,tag=sys.argv[1],Path(sys.argv[2]),sys.argv[3]
refs=S.read(S.SETS/f'{S.TSET}.jsonl');rows={r['id']:r for r in S.read(rundir/f'{tag}-0-{S.N}.jsonl')};assert all(r['id'] in rows for r in refs),tag
d=rundir/'eval'/tag;resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
assert json.loads((d/'responses.json').read_text())==resp,f'{d}/responses.json differs from the run file: not judged'
names=sorted(p.name for p in (d/'combined').glob('*.json'));assert any('flan-t5-base' in n for n in names) and any('flan-t5-large' in n for n in names),names
if gpu=='--check':print('check ok',rundir.name,tag,names);sys.exit(0)
if not (d/'combined.bak-1108').exists():shutil.copytree(d/'combined',d/'combined.bak-1108')
S.judge(gpu,d);print('judged xl',rundir.name,tag,flush=True)
