import re,subprocess,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');DEV=O/'runs/gemma_throne_dev60'
CAND={'s24a1bl2p1':'--alpha 1 --layer 24 --lam 2 --pen 1','s20a1bl2p1':'--alpha 1 --layer 20 --lam 2 --pen 1',
      's28a1bl2p1':'--alpha 1 --layer 28 --lam 2 --pen 1','bl2p1':'--lam 2 --pen 1'}
out=subprocess.run([sys.executable,'-m','tether.eval.throne_partial',str(DEV),'base',*CAND],
                   capture_output=True,text=True,check=True).stdout
print(out,end='')
f={m[1]:float(m[2]) for m in re.finditer(r'^(\S+)\s+P .*F0\.5\s+([\d.]+)',out,re.M)}
assert set(CAND)<=set(f),f
pick=max(CAND,key=lambda t:(f[t],-list(CAND).index(t)))
(O/'queues/gemma_ours.txt').write_text(f'# dev pick {pick} (F0.5 {f[pick]:.1f}, base {f["base"]:.1f})\n'+''.join(
    f'vdd throne_full1000 gemma_throne1000_parts {pick} {b} {b+50} {CAND[pick]}\n' for b in range(0,1000,50)))
print('picked',pick)
