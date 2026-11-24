"""고정 alpha 게이트의 fold별, 모델별 변화와 Sinkhorn과의 겹침. 사용법: python check_gate.py <repo> [famsvc npz | -] [OOF npz] (인자는 gate2.py와 같음)"""
import sys, numpy as np
from pathlib import Path
exec(open(Path(__file__).parent / "gate2.py").read().split("res = {}")[0])  # gate2.py의 함수와 데이터 로드만 재사용
Ss=[spec(n,c,[ng[:,c],lo[:,c]],1.0) for n,c in FAMS.items()]
AL=[0.75,0.25,0.5,1.0]
g=base
for c,S,a in zip(FAMS.values(),Ss,AL): g=gate(g,c,S,a)
def sink(lp,rho=.35,it=10,dmp=.5):
    sm=lambda z: np.exp(z-logsumexp(z,1,keepdims=True))
    t=(1-rho)*sm(lp).mean(0)+rho/lp.shape[1]; b=np.zeros(lp.shape[1])
    for _ in range(it): b+=dmp*(np.log(t)-np.log(sm(lp+b).mean(0)))
    return lp+b
print('alpha 고정(전체OOF 선택) 게이트 BA', round(ba(g),4), '| +Sinkhorn: v4', round(ba(sink(base)),4), '게이트', round(ba(sink(g)),4))
for k in range(5):
    m=fold==k; print(' fold',k, round(ba(base,m),4),'->',round(ba(g,m),4))
rb=np.array([(base.argmax(1)[y==k]==k).mean() for k in range(K)]); rg=np.array([(g.argmax(1)[y==k]==k).mean() for k in range(K)])
for k in np.argsort(rg-rb): print(f'  {L[k]:24s} {rb[k]:.3f} -> {rg[k]:.3f} ({rg[k]-rb[k]:+.3f})')
