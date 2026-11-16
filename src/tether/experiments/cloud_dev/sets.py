import concurrent.futures as cf,hashlib,json,sys,time,urllib.request
from pathlib import Path
from PIL import Image

R=Path('/home/project-b/LJW/vlm-grounding-steering');W=Path('/workspace');S=W/'sets'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,rows):Path(p).write_text(''.join(json.dumps(r)+'\n' for r in rows))

def get(it):
    p=Path(it['image'])
    for k in range(8):
        try:
            if not p.exists():
                data=urllib.request.urlopen(it['url'],timeout=60).read();tmp=p.with_suffix('.part');tmp.write_bytes(data);tmp.rename(p)
            with Image.open(p) as im:im.convert('RGB')
            return p.name
        except Exception:
            p.unlink(missing_ok=True);time.sleep(2+k)
    raise RuntimeError(f'failed {it["url"]}')

def images():
    items=read(R/'data/coco-train2014-probe5000/list.jsonl')[:100]
    with cf.ThreadPoolExecutor(16) as ex:done=list(ex.map(get,items))
    print('dev images ok',len(done))

def manifests():
    S.mkdir(exist_ok=True)
    write(S/'chair_dev60.jsonl',read(R/'data/chair-dev-train2014-100/chair.jsonl')[:60])
    write(S/'chair_test60.jsonl',read(W/'chair_test100.jsonl')[:60])
    a=read(W/'amber100.jsonl');write(S/'amber_test60.jsonl',a[:60])
    root=Path(a[0]['image']).parent;dev=[]
    for i in range(101,161):
        p=root/f'AMBER_{i}.jpg'
        dev.append({**a[0],'id':f'amber:{i}','group_id':f'amber:{i}','split':'dev','image':str(p),
                    'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metadata':{'image_id':i,'amber_id':i}})
    write(S/'amber_dev60.jsonl',dev)
    for f in sorted(S.glob('*.jsonl')):print(f.name,len(read(f)))

if __name__=='__main__':{'images':images,'manifests':manifests}[sys.argv[1]]()
