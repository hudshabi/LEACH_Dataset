"""Independent batch check of retry formula in an affordable one-round snapshot.
Checks moments, not H-round survival approximation or policy optimality.
"""
import json,numpy as np
from pathlib import Path
from extension_protocol import ExtensionEnv,O
from independent_batch_simulator import IndependentBatch

def main():
 e=ExtensionEnv(40,optimized=True,H=20);e.reset(seed=9900);heads=e.ch.copy();D=0.;E=0.;members=[i for i in range(100) if i not in heads]
 for c in heads:
  incoming=0.;rx=0.
  for i in members:
   h=heads[np.argmin(e.dist[i,heads])];d=e.dist[i,h]
   if h==c and d<=70:
    incoming+=e.pm[i,c];rx+=.0002*e.am[i,c];E+=e.am[i,c]*(.0002+4e-7*d*d)
  D+=e.pb[c]*(1+incoming);E+=rx+e.ab[c]*(.0002+4e-7*e.dbs[c]**2)
 snapshot={k:getattr(e,k) for k in ['pos','e','states','last','ch','t','age','prev','prev_pdr','setup']};snapshot['setup']=False;snapshot['e']=np.full(100,100.)
 # Randomize hidden states from stationarity, then evolve once using independent dynamics.
 b=IndependentBatch.from_snapshot(snapshot,list(range(100000,110000)));b.states=np.random.default_rng(4305).random((10000,100,2))<.625
 v=b.step(np.zeros(10000,int));out={}
 for name,pred,vals in [('delivery',D,v['delivered']),('energy',E,v['energy'])]:
  mean=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)));out[name]=dict(predicted=pred,mc_mean=mean,standard_error=se,z_error=(mean-pred)/se,passed=bool(abs(mean-pred)<4*se))
 assert all(v['passed'] for v in out.values()),out
 out.update(replicates=10000,condition='nominal',scope='Stationary one-round affordable transmissions only; no validation of H-round survival approximation')
 (O/'forecast_mc_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
