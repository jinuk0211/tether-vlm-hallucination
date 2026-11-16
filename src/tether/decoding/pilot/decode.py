import argparse,json,math,os,time,traceback
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList
from tether.models.backend import load_adapter
from tether.analysis.read_gap import collect as C
from tether.decoding.common import ROOT, r_curve

class State:
    def __init__(self):self.t=0;self.on=False

class Belief(LogitsProcessor):
    def __init__(self,st,vidx,rf,lam,clip,prompt_len,uniform=False):
        self.st=st;self.vidx=vidx;self.rf=rf;self.lam=lam;self.clip=clip;self.pl=prompt_len;self.uniform=uniform
        self.A=torch.zeros(len(vidx),device=vidx.device);self.logV=math.log(len(vidx));self.trace=[]
    def __call__(self,input_ids,scores):
        t=input_ids.shape[1]-self.pl;self.st.t=t+1
        if self.lam:
            lp=scores[0].float().log_softmax(-1)[self.vidx]
            self.A+=self.rf(t)*lp
            adj=self.lam*(self.A-torch.logsumexp(self.A,0)+self.logV).clamp(-self.clip,self.clip)
            done=torch.isin(self.vidx,input_ids[0,self.pl:])
            if self.uniform:adj=adj[~done].mean().expand_as(adj).clone() if (~done).any() else adj
            adj=torch.where(done,torch.zeros_like(adj),adj)
            scores[0,self.vidx]+=adj.to(scores.dtype)
        return scores

class Steer:
    def __init__(self,layer,st,d,alpha,rf):
        self.st=st;self.d=d;self.alpha=alpha;self.rf=rf;layer.register_forward_hook(self.hook)
    def hook(self,mod,args,out):
        h=out[0] if isinstance(out,tuple) else out
        if not self.st.on or h.shape[1]!=1 or not self.alpha:return None
        h=h+(self.alpha*(1-self.rf(self.st.t))*self.d).to(h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    refs=C.read(a.data)[a.start:a.end]
    name=f'{a.tag}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    done=C.read(rows_file) if rows_file.exists() else []
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),args={k:str(v) for k,v in vars(a).items()},resumed_from=len(done))
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer
        vobj=json.loads((ROOT/f'outputs/rbd-20261030/collect/{a.model}-vobj2.json').read_text())
        vidx=torch.tensor(vobj,device=adapter.device);rf=r_curve(a.model,a.r)
        st=State()
        if a.alpha:
            d=torch.load(ROOT/f'outputs/rbd-20261030/dvec/{a.model}-dvec.pt',weights_only=False)['d'][a.layer].to(adapter.device)
            Steer(adapter.layers[a.layer-1],st,d,a.alpha,rf)
        update(stage='running',vobj=len(vobj))
        with rows_file.open('a') as f:
            for i,ref in enumerate(refs[len(done):],start=len(done)):
                t0=time.perf_counter();batch=adapter.prepare(ref['image'],ref['prompt']);pl=batch['input_ids'].shape[1]
                proc=Belief(st,vidx,rf,a.lam,a.clip,pl,a.uniform);st.t=0;st.on=True
                try:
                    with torch.no_grad():
                        o=m.generate(**{k:v for k,v in batch.items() if k!='labels'},do_sample=False,num_beams=1,use_cache=True,
                                     max_new_tokens=int(ref['max_new_tokens']),logits_processor=LogitsProcessorList([proc]))
                finally:st.on=False
                ids=o[0,pl:].tolist();text=tok.decode(ids,skip_special_tokens=True)
                f.write(json.dumps({**ref,'method':a.tag,'prediction':text,'generated_tokens':len(ids),'elapsed_seconds':time.perf_counter()-t0,
                                    'lam':a.lam,'alpha':a.alpha,'layer':a.layer,'r':a.r,'clip':a.clip})+'\n');f.flush()
                update(done=i+1)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(C.MODELS),required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/chair-dev-train2014-100/chair.jsonl')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=100)
    p.add_argument('--lam',type=float,default=0.0);p.add_argument('--clip',type=float,default=10.0)
    p.add_argument('--alpha',type=float,default=0.0);p.add_argument('--layer',type=int,default=16)
    p.add_argument('--r',choices=['measured','const','step0'],default='measured')
    p.add_argument('--uniform',action='store_true',help='control: replace the per-object belief push by its mean over V_obj')
    main(p.parse_args())
