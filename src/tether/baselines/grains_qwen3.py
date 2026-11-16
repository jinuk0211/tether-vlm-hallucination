import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tether.baselines.author_bridge import load_authors
from tether.baselines.qwen_compat import QwenAttributionAPI
from tether.models.backend import load_adapter
from tether.cli import _seed
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering');OUT=ROOT/'outputs/qwen3-vl8b';X=OUT/'grains-extraction';X.mkdir(parents=True,exist_ok=True)
FIT=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/data/fit50.jsonl'
def main(limit):
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    config=json.loads((ROOT/'configs/qwen3_vl_8b.json').read_text());_seed(config)
    a=load_adapter(dict(config['model'],dtype='float32',attn_implementation='eager'));assert a.model_type=='qwen3_vl' and len(a.layers)==36
    for p in a.model.parameters():p.requires_grad_(False)
    authors=load_authors();fit=[json.loads(s) for s in FIT.read_text().splitlines() if s.strip()];assert len(fit)==50
    for i,r in enumerate(fit[:limit]):
        f=X/f'{i:03d}.npz'
        if f.exists():continue
        _seed(config);t=time.perf_counter()
        with Image.open(r['image']) as s:image=s.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
        prompt=a.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':r['prompt']}]}],tokenize=False,add_generation_prompt=True)
        v=authors.ablation(QwenAttributionAPI(a.model),a.processor,image,prompt,r['chosen'],r['rejected'],top_k=50,top='pos',attribution_method='integrated_gradients')
        for k,x in v.items():assert np.isfinite(x).all(),k
        tmp=X/f'{i:03d}.tmp.npz';np.savez(tmp,**v);tmp.replace(f)
        print(f'{i+1}/50 {time.perf_counter()-t:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB',flush=True)
    if limit<50:print('LIMIT_DONE',flush=True);return
    H={}
    for i in range(50):
        with np.load(X/f'{i:03d}.npz') as z:
            for k in z.files:
                if k.startswith('hidden_'):H.setdefault(k,[]).append(z[k])
    assert len(H)==4;arrays={k:np.stack(v) for k,v in H.items()};_seed(config)
    vec=authors.steering.compute_steering_vector(arrays,27,'pca');assert torch.isfinite(vec).all() and vec.norm()>0
    torch.save({'vector':vec.cpu(),'layer':27,'alpha':10.0,'fit_records':[r['id'] for r in fit],'model':config['model'],'ig_steps':20,'top_k':50,'mode':'pos','vector_method':'pca'},OUT/'grains-author.pt')
    print('GRAINS_VECTOR_DONE norm',float(vec.norm()),'dtype',vec.dtype,flush=True)
if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 50)
