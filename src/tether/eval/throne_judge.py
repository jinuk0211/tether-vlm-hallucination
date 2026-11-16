import json,os,shutil,subprocess,sys
from pathlib import Path
gpu,M,tag,b,J=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
assert J in ('base','large','xl'),J
BENCH=Path(os.environ.get('BENCH','/home/project-b/LJW/bench'))
T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json');JUDGE=str(BENCH/f'models/flan-t5-{J}')
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
refs=read(O/'sets/throne_full.jsonl')[1000*b:1000*(b+1)];assert len(refs)==1000
rows={}
for f in sorted((O/'runs'/f'throne_parts_{M}').glob(f'{tag}-*-*.jsonl')):
    if f.name[:-6].rsplit('-',2)[0]!=tag:continue
    for r in read(f):rows[r['id']]=r
miss=[r['id'] for r in refs if r['id'] not in rows];assert not miss,f'{len(miss)} rows of block {b} not decoded'
assert all(rows[r['id']]['image_sha256']==r['image_sha256'] for r in refs)
d=O/'runs'/f'throne_{M}'/'eval'/tag/f'b{b}-{J}';d.mkdir(parents=True,exist_ok=True)
resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
if not (d/'responses.json').exists() or json.loads((d/'responses.json').read_text())!=resp:
    shutil.rmtree(d/'combined',ignore_errors=True);(d/'responses.json').write_text(json.dumps(resp))
(d/'combined').mkdir(exist_ok=True);name=f"evaluations_{JUDGE.replace('/','_')}.json"
if not (d/'combined'/name).exists():
    env={**os.environ,'LOCAL_RANK':'0','WORLD_SIZE':'1','RANK':'0','CUDA_VISIBLE_DEVICES':gpu}
    out=subprocess.run([sys.executable,'throne_aqa_evaluation.py','--coco_file',COCO,'--response_file',str(d/'responses.json'),
                        '--evaluator_model_path',JUDGE,'--per_device_batch_size','32','--save_path',str(d/'evaluations.json')],
                       cwd=T,env=env,capture_output=True,text=True)
    assert out.returncode==0,out.stderr[-3000:]
    shutil.move(d/name,d/'combined'/name)
print('judged',M,tag,b,J,flush=True)
