import hashlib,json
from pathlib import Path
B=Path('/home/project-b/LJW/bench/coco2017');S=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/sets')
inst=json.load(open(B/'annotations/instances_val2017.json'));ids=sorted(i['id'] for i in inst['images'])
rows=[]
for i in ids:
    p=B/'val2017'/f'{i:012d}.jpg'
    rows.append({'id':f'throne:{i}','group_id':f'throne:{i}','task':'caption','benchmark':'throne_coco','split':'test','image':str(p),
                 'prompt':'Describe this image in detail.','max_new_tokens':512,
                 'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metadata':{'image_id':i}})
assert len(rows)==5000
(S/'throne_full.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('throne_full',len(rows),'ids',ids[0],'..',ids[-1])
