import json,os,random,sys,time,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering');U=ROOT/'upstream/Revis';OUT=ROOT/'outputs/qwen3-vl8b/revis-official';IMG=ROOT/'data/revis-official/images';OUT.mkdir(parents=True,exist_ok=True)
from tether.models.backend import load_adapter
def load():
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    c=json.loads((ROOT/'configs/qwen3_vl_8b.json').read_text());a=load_adapter(dict(c['model'],dtype='bfloat16',attn_implementation='sdpa'))
    assert a.model_type=='qwen3_vl' and len(a.layers)==36;return a
def vectors(a):
    sys.path[:0]=[str(U),str(ROOT/'.author-deps')]
    sys.modules['qwen_vl_utils']=types.SimpleNamespace(process_vision_info=None)
    from steering_utils.vector_calculator import _calculate_das_vectors_internal as calc
    m=a.model
    def to_device(_m,args,kw):return args,{k:(x.to(a.device) if torch.is_tensor(x) else x) for k,x in kw.items()}
    pre=m.register_forward_pre_hook(to_device,with_kwargs=True)
    if not hasattr(m,'language_model') and getattr(m.config,'num_hidden_layers',None) is None:
        m.config.num_hidden_layers=len(a.layers);print('num_hidden_layers set on the top config:',len(a.layers),flush=True)
    def run(name):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        with torch.no_grad():x=calc(m,a.processor,str(U/'data/OH_sampled_data_100.json'),str(IMG),name)
        assert sorted(x)==list(range(37)),sorted(x);return x
    v=run('Qwen');w=run('Qwen3');pre.remove()
    torch.save(v,OUT/'vectors.pt');torch.save(w,OUT/'vectors-qwen3-branch.pt')
    print('vectors',len(v),{k:round(float(v[k]['visual'].norm()),1) for k in (20,28,32,36)},'cos with the Qwen3 branch',{k:round(float(F.cosine_similarity(v[k]['visual'],w[k]['visual'],dim=0)),3) for k in (20,28,32,36)},flush=True)
def calibrate(a,k=80.0,min_layer=10,n=100):
    v=torch.load(OUT/'vectors.pt',map_location='cpu',weights_only=True);m=a.model;p=a.processor
    Q=[json.loads(l) for l in open(U/'data/coco_pope_calibration_merged.jsonl') if l.strip()][:n];res={}
    for L in range(len(a.layers)-1,min_layer-1,-1):
        vec=v[L+1]['visual'];cap=[None]
        def hook(_m,_i,out):
            h=out[0] if isinstance(out,tuple) else out
            if cap[0] is None:
                x=F.normalize(h[:,-1,:],dim=-1);u=F.normalize(vec.to(device=h.device,dtype=h.dtype),dim=-1);cap[0]=-torch.matmul(x,u).item()
            return out
        H=a.layers[L].register_forward_hook(hook);rec=[]
        for it in Q:
            cap[0]=None;im=Image.open(IMG/it['image']).convert('RGB')
            msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':it['text']+' Please answer yes or no.'}]}]
            txt=p.apply_chat_template(msg,add_generation_prompt=True,tokenize=False);inp=p(text=txt,images=[im],padding=True,return_tensors='pt').to(m.device)
            with torch.no_grad():o=m.generate(**inp,max_new_tokens=5,do_sample=False)
            g=p.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip().lower();pred='yes' if 'yes' in g else ('no' if 'no' in g else 'unknown');gt=it['label'].lower()
            c='TP' if gt=='yes' and pred=='yes' else 'FP' if gt=='no' and pred=='yes' else 'TN' if gt=='no' and pred=='no' else 'Other';rec.append((cap[0],c))
        H.remove()
        fa=[r for r,c in rec if c in('TP','TN')];ha=[r for r,c in rec if c=='FP']
        print(f'layer {L}: factual {len(fa)} hallucinated {len(ha)} other {len(rec)-len(fa)-len(ha)}'+(f' sep {np.mean(ha)-np.mean(fa):+.4f} tau {np.percentile(fa,k):+.4f}' if fa and ha else ''),flush=True)
        res[L]=dict(factual=len(fa),hallucinated=len(ha),separability=float(np.mean(ha)-np.mean(fa)) if fa and ha else None,tau=float(np.percentile(fa,k)) if fa else None)
        if fa and ha and np.mean(ha)-np.mean(fa)>0:
            (OUT/'calibration.json').write_text(json.dumps(dict(layer=L,tau=res[L]['tau'],k=k,alpha=1.6,separability=res[L]['separability'],search=res,time=time.strftime('%F %T')),indent=1));print('L*',L,'tau',res[L]['tau'],flush=True);return
    raise SystemExit('no layer has positive separability (or no false positives in the calibration rows)')
if __name__=='__main__':
    a=load()
    if not (OUT/'vectors.pt').exists():vectors(a)
    calibrate(a)
    print('REVIS_ARTIFACTS_DONE',flush=True)
