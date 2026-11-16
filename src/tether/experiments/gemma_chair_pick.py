import json,sys
s=json.load(open(sys.argv[1]))['results'];b=s['base'];c={t:v for t,v in s.items() if t!='base'}
ok=[t for t,v in c.items() if v['recall']>=b['recall']-2]
f1=lambda v:2*(100-v['CHAIR_i'])*v['recall']/(100-v['CHAIR_i']+v['recall'])
print(min(ok,key=lambda t:(c[t]['CHAIR_s'],c[t]['CHAIR_i'])) if ok else max(c,key=lambda t:f1(c[t])))
