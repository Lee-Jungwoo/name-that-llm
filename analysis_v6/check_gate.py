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

# 고정 alpha 이득의 프롬프트 bootstrap, 고친/망친 지문 수, 회사 하나만 게이트했을 때, 다른 alpha 조합
def ba_idx(lp,ii):
    yy,p=y[ii],lp[ii].argmax(1); return np.mean([(p[yy==k]==k).mean() for k in range(K)])
rng=np.random.default_rng(0); P=np.unique(pid); idx_by_p={p:np.where(pid==p)[0] for p in P}; diffs=[]
for _ in range(1000):
    ii=np.concatenate([idx_by_p[p] for p in rng.choice(P,len(P),replace=True)]); diffs.append(ba_idx(g,ii)-ba_idx(base,ii))
lo_,hi_=np.percentile(diffs,[2.5,97.5]); pb,pg=base.argmax(1),g.argmax(1)
print(f'고정 alpha dBA {ba(g)-ba(base):+.4f}, 프롬프트 bootstrap 95% [{lo_:+.4f}, {hi_:+.4f}], dBA>0 비율 {np.mean(np.array(diffs)>0):.3f} | 바뀐 지문 {(pb!=pg).sum()} (고침 {((pg==y)&(pb!=y)).sum()}, 망침 {((pg!=y)&(pb==y)).sum()})')
for (n,c),S,a0 in zip(FAMS.items(),Ss,AL):
    print(f'  {n:7s}만 게이트 (고정 alpha {a0}: {ba(gate(base,c,S,a0))-ba(base):+.4f}) | alpha별 '+', '.join(f'{a}: {ba(gate(base,c,S,a))-ba(base):+.4f}' for a in (.25,.5,.75,1.0)))
for name,al in [('고정',AL),('모두 0.25',[.25]*4),('모두 0.5',[.5]*4),('모두 0.75',[.75]*4),('모두 1.0',[1.0]*4)]:
    t=base
    for c,S,a in zip(FAMS.values(),Ss,al): t=gate(t,c,S,a)
    print(f'  alpha {name:9s}: {ba(t):.4f} ({ba(t)-ba(base):+.4f}) | fold별 '+' '.join(f'{ba(t,fold==k)-ba(base,fold==k):+.4f}' for k in range(5)))
