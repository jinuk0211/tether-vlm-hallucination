import json,os,shutil,sys
from pathlib import Path
from tether.eval import throne as S
if os.environ.get('RBD2_JUDGES'):S.JUDGES=[str(S.BENCH/f'models/flan-t5-{s}') for s in os.environ['RBD2_JUDGES'].split(',')]
gpu,rundir,tags=sys.argv[1],Path(sys.argv[2]),sys.argv[3:]
refs=S.read(S.SETS/f'{S.TSET}.jsonl');ids=[r['id'] for r in refs]
for tag in tags:
    rows={r['id']:r for r in S.read(rundir/f'{tag}-0-{S.N}.jsonl')};assert all(i in rows for i in ids),tag
    d=rundir/'eval'/tag;d.mkdir(parents=True,exist_ok=True)
    resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
    if not (d/'responses.json').exists() or json.loads((d/'responses.json').read_text())!=resp:
        shutil.rmtree(d/'combined',ignore_errors=True);(d/'responses.json').write_text(json.dumps(resp))
    S.judge(gpu,d);print('judged',tag,flush=True)
