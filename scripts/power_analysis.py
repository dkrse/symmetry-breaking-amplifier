#!/usr/bin/env python3

import numpy as np
from symmetry_breaking import step

# ---------- P1: decoupling rises with gain (Music Lab) ----------
def market(g, Nsongs=48, Ntrials=700, seed=0):
    r=np.random.default_rng(seed)
    q=r.uniform(0.3,1.0,Nsongs)
    d=np.ones(Nsongs)
    for _ in range(Ntrials):
        share=d/d.sum()
        appeal=q*(1+g*share*Nsongs)
        d[r.choice(Nsongs,p=appeal/appeal.sum())]+=1
    return q,d

def spearman(a,b):
    ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b))
    return np.corrcoef(ra,rb)[0,1]

gains=[0.0,0.5,1.0,2.0,4.0]
W=30
C=np.zeros((len(gains),W))

for i,g in enumerate(gains):
    for w in range(W):
        q,d=market(g,seed=100*i+w); C[i,w]=spearman(q,d)
print('P1 corr(outcome, quality) by gain:')

for g,row in zip(gains,C): print(f'  g={g:4.1f}  corr={row.mean():.3f} +/- {row.std():.3f}')

# power: slope of corr on gain < 0
def exp1(W,seed):
    r=np.random.default_rng(seed);xs=[];ys=[]
    for i,g in enumerate(gains):
        for w in range(W):
            q,d=market(g,seed=r.integers(1e9));xs.append(g);ys.append(spearman(q,d))
    xs=np.array(xs);ys=np.array(ys)
    A=np.vstack([np.ones_like(xs),xs]).T
    b,*_=np.linalg.lstsq(A,ys,rcond=None)
    resid=ys-A@b;s2=resid@resid/(len(ys)-2);cov=s2*np.linalg.inv(A.T@A)
    return b[1]/np.sqrt(cov[1,1])

for W in [10,20,30]:
    ts=[exp1(W,s) for s in range(50)]
    print(f'  P1 power W={W}: {np.mean(np.array(ts)<-1.98):.2f}')

# ---------- P8: cohort concentration (Gini), high-gain vs low-gain domain ----------
# We score the domain contrast by the Gini of final outcomes, NOT a log-gap or a
# mean/median ratio: on heavy-tailed outcomes the log compresses the tail that
# carries the most amplified domain, and the ratio diverges as the median
# collapses (median -> 0 at high gain). The Gini stays bounded in [0,1] and
# monotone in gain. See Section 'First real-data evidence'.
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x); c=np.cumsum(x)
    return float((n+1-2*np.sum(c)/c[-1])/n)

def cohort_gini(g,M=2000,T=300,seed=0):
    r=np.random.default_rng(seed);x=r.normal(0,1e-6,M)
    rk=dict(p_hi=0.01,S_crit=200.0,w=0.5)
    for _ in range(T): x=step(x,r,g,0.05,0.02,50.0,rk)
    S=np.exp(x-x.max())            # raw status (rescaled; Gini is scale-invariant)
    return gini(S)

gl,gh=0.05,0.4  # low- vs high-amplification domain
print('\nP8 concentration (Gini of final outcomes):')
for name,g in [('low-gain',gl),('high-gain',gh)]:
    vals=[cohort_gini(g,seed=s) for s in range(30)]
    print(f'  {name} (g={g}): Gini={np.mean(vals):.3f} +/- {np.std(vals):.3f}')

def exp8(n,seed):
    r=np.random.default_rng(seed)
    lo=[cohort_gini(gl,seed=r.integers(1e9)) for _ in range(n)]
    hi=[cohort_gini(gh,seed=r.integers(1e9)) for _ in range(n)]
    lo=np.array(lo);hi=np.array(hi)
    t=(hi.mean()-lo.mean())/np.sqrt(hi.var(ddof=1)/n+lo.var(ddof=1)/n)

    return t

for n in [10,20,30]:
    ts=[exp8(n,s) for s in range(40)]
    print(f'  P8 power n={n} cohorts/domain: {np.mean(np.array(ts)>1.98):.2f}')

# ---------- Withdrawn prediction: a free-token market has NO optimal-gain downturn ----------
# The download market (preferential attachment, no per-unit ceiling) concentrates
# monotonically in the social-signal strength g -- no non-monotone peak, because it
# has no S* (see Section 'empirical', the withdrawn prediction). Reproduces the
# Gini sweep quoted there.
print('\nDownload-market concentration (Gini of final download counts) by gain:')
for g in [0.0,0.5,1.0,2.0,4.0,8.0]:
    gv=np.mean([gini(market(g,seed=100+w)[1]) for w in range(10)])
    print(f'  g={g:4.1f}  Gini={gv:.3f}')
