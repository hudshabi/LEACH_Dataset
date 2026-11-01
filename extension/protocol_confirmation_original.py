"""Prospective model-based extension. No instantaneous channel or future-draw access.
Local head-swap optimizer; analytic planning approximation, not MILP or RL.
"""
import os,time,json,hashlib
from pathlib import Path
import numpy as np
from run_equal_budget import EnergyBudgetEnv
from base_simulator import TwoHopEnv
from run_study import write_rows
R=Path(__file__).resolve().parent; O=R/'extension'
class ExtensionEnv(EnergyBudgetEnv):
 def __init__(self,budget=40,condition='nominal',optimized=False,H=10,retry=True,survival=True):
  self.condition=condition;self.optimized=optimized;self.H=H;self.retry_model=retry;self.survival_model=survival;self.solve_calls=0;self.candidate_evaluations=0
  super().__init__(budget,False,False)
 def reset(self,**kw):
  self.prepared=False;obs,info=super().reset(**kw)
  if self.condition=='far_sink':self.dbs=np.linalg.norm(self.pos-[50,225],axis=1)
  if self.condition=='heterogeneous':
   # Separate deterministic stream leaves exogenous channel stream unchanged.
   en=np.random.default_rng(int(kw['seed'])+900000).uniform(.25,.75,100);self.e=50*en/en.sum()
  self.prepared=True;self.control_debit=.00001+1e-8*self.dbs**2
  pg=(.1+.8*np.exp(-2.5*self.dist/200))*(.7 if self.condition=='poor_channel' else 1.)
  pb=(.1+.8*np.exp(-2.5*self.dbs/200))*(.7 if self.condition=='poor_channel' else 1.)
  self.pm,self.am=self.moments(pg);self.pb,self.ab=self.moments(pb)
  self.ch=self.cluster();self.telemetry_bits=0;self.telemetry_energy=0.;self.changes=0;self.optional_requests=0;self.decisions=[]
  return self.obs(),info
 def moments(self,p):
  if not self.retry_model:return .625*p+.375*.3*p,np.ones_like(p)
  success=.625*(1-(1-p)**3)+.375*(1-(1-.3*p)**3)
  attempts=.625*(1+(1-p)+(1-p)**2)+.375*(1+(1-.3*p)+(1-.3*p)**2)
  return success,attempts
 def prob(self,d,state):return super().prob(d,state)*(.7 if self.condition=='poor_channel' else 1.)
 def cluster(self):
  if not self.prepared or not self.optimized:return TwoHopEnv.cluster(self)
  return self.optimize()
 def forecast(self,sets,reform=False):
  """Vectorized deterministic H-round yield surrogate over candidate head sets."""
  sets=np.asarray(sets,dtype=int);sets=sets[None,:] if sets.ndim==1 else sets
  B,k=sets.shape;alive=self.e>1e-12
  if not k:return np.zeros(B)
  d=self.dist[:,sets].transpose(1,0,2) # B,N,k
  nearest=np.argmin(d,axis=2);distance=np.take_along_axis(d,nearest[:,:,None],axis=2)[:,:,0]
  member=np.broadcast_to(alive,(B,100)).copy();member[np.arange(B)[:,None],sets]=False
  valid=member&(distance<=70)
  chosen=sets[np.arange(B)[:,None],nearest]
  sm=self.pm[np.arange(100)[None,:],chosen]*valid;am=self.am[np.arange(100)[None,:],chosen]*valid
  member_load=am*(.0002+4e-7*distance**2)
  loads=member_load.copy()
  rx=np.zeros((B,k))
  for j in range(k):rx[:,j]=np.sum(am*(nearest==j),axis=1)*.0002
  headload=rx+self.ab[sets]*(.0002+4e-7*self.dbs[sets]**2)
  loads[np.arange(B)[:,None],sets]=headload
  available=np.maximum(0,self.e-(self.control_debit if reform else 0))
  if self.survival_model:u=np.minimum(1,available[None,:]/np.maximum(self.H*loads,1e-15))
  else:u=np.broadcast_to(alive,(B,100)).astype(float)
  Hdelivery=np.zeros(B)
  for j in range(k):
   # Shared head lifetime bounds own and aggregated member contribution.
   member_service=np.minimum(u,u[np.arange(B),sets[:,j]][:,None])
   Hdelivery+=self.H*self.pb[sets[:,j]]*(u[np.arange(B),sets[:,j]]+np.sum(sm*(nearest==j)*member_service,axis=1))
  expenditure=np.minimum(available[None,:],self.H*loads).sum(axis=1)
  if reform:expenditure+=np.minimum(self.e,self.control_debit).sum()
  return Hdelivery/np.maximum(expenditure,1e-12)
 def optimize(self):
  self.solve_calls+=1
  base=TwoHopEnv.cluster(self);alive=np.flatnonzero(self.e>1e-12)
  if len(alive)<=len(base):return base
  # Candidate pool combines efficient BS forwarding and spatial coverage.
  merit=self.e/(.0002+4e-7*self.dbs**2)
  top=alive[np.argsort(-merit[alive],kind='stable')[:8]]
  spatial=[]
  for x in [25,75]:
   for y in [25,75]:
    scores=self.e[alive]/(.1+np.linalg.norm(self.pos[alive]-[x,y],axis=1)/100)
    spatial.extend(alive[np.argsort(-scores,kind='stable')[:2]])
  incumbent=np.sort(base);current=getattr(self,'ch',base)
  pool=np.unique(np.concatenate([top,spatial,base,current]));pool=pool[self.e[pool]>1e-12]
  for sweep in range(2):
   options=[incumbent]
   for j in range(len(incumbent)):
    for c in pool:
     if c not in incumbent:
      z=incumbent.copy();z[j]=c;options.append(np.sort(z))
   options=np.unique(np.array(options),axis=0);self.candidate_evaluations+=len(options)
   scores=self.forecast(options,True);idx=int(np.argmax(scores));new=options[idx]
   if np.array_equal(incumbent,new):break
   incumbent=new
  return incumbent
 def poll(self,policy):
  if self.setup or not len(self.ch) or np.any(self.e[self.ch]<=1e-12):return 0
  if self.t%5:return 0
  # Extra status poll is charged even if reform is subsequently rejected.
  before=self.e.sum();alive=np.flatnonzero(self.e>1e-12)
  for i in alive:self.consume(i,.000005+1e-8*self.dbs[i]**2)
  energy=float(before-self.e.sum());self.totals['energy']+=energy;self.totals['overhead']+=100*len(alive);self.telemetry_energy+=energy;self.telemetry_bits+=100*len(alive)
  forced=bool(np.any(self.e[self.ch]<=1e-12))
  candidate=self.cluster();old=self.forecast(self.ch,False)[0];new=self.forecast(candidate,True)[0]
  action=int(new>old*1.02 and not np.array_equal(np.sort(self.ch),np.sort(candidate)))
  self.decisions.append(dict(round=self.t,old_yield=old,new_yield=new,poll_energy=energy,request=action,forced_after_poll=int(forced)))
  self.optional_requests+=int(action and not forced)
  return action

def episode(policy,scenario,condition,budget,H=10,period=5,trace=False):
 optimized=policy in ['OptimizedPeriod','OptimizedNever','OptimizedGate','NoRetryGate','NoSurvivalGate','NoRetryNever','NoSurvivalNever']
 gate=policy in ['BaseGate','OptimizedGate','NoRetryGate','NoSurvivalGate']
 e=ExtensionEnv(budget,condition,optimized,H,policy not in ['NoRetryGate','NoRetryNever'],policy not in ['NoSurvivalGate','NoSurvivalNever']);obs,_=e.reset(seed=scenario)
 start=time.perf_counter();requests=changed=0;rows=[];T10=T50=None;c200=0
 while e.used<budget-1e-10:
  prior=e.ch.copy()
  if gate:action=e.poll(policy)
  elif policy in ['Never','OptimizedNever','NoRetryNever','NoSurvivalNever']:action=0
  else:action=int(e.t%(5 if policy=='Period5' else period)==0)
  if e.used>=budget-1e-10:break
  obs,r,t,x,info=e.step(action);requests+=action;different=int(not np.array_equal(np.sort(prior),np.sort(e.ch)));changed+=different
  if e.t<=200:c200+=info['delivered']
  if T10 is None and info['alive']<=90:T10=e.t
  if T50 is None and info['alive']<=50:T50=e.t
  if trace:rows.append(dict(round=e.t,spent=e.used,changed=different,heads=','.join(map(str,e.ch)),**{k:v for k,v in info.items() if k!='round'}))
  if t or x:break
 assert abs(e.e.sum()+e.used-50)<1e-8
 assert e.used<=budget+1e-9
 z=e.totals
 row=dict(policy=policy,scenario=scenario,condition=condition,budget=budget,H=H,period=period,spent=e.used,delivered=z['delivered'],rounds=e.t,alive_final=int((e.e>1e-12).sum()),generated=z['generated'],PDR=z['delivered']/max(1,z['generated']),mJ_per_sample=1000*e.used/max(1,z['delivered']),control_bits=z['overhead'],telemetry_bits=e.telemetry_bits,telemetry_energy=e.telemetry_energy,requests=requests,changed=changed,reclusters=z['reclusters'],optional_requests=e.optional_requests,solve_calls=e.solve_calls,candidate_evaluations=e.candidate_evaluations,runtime=time.perf_counter()-start,T10=T10 if T10 else '',T50=T50 if T50 else '',C200=c200/20000 if e.t>=200 else '',budget_reached=int(e.used>=budget-1e-9))
 return row,rows,e.decisions

def job(args):
 policy,scenario,condition,budget,H,period,stage=args
 row,trace,decisions=episode(policy,scenario,condition,budget,H,period,trace=True)
 dest=O/stage/f'{policy}_{condition}_{scenario}_{budget}_H{H}_P{period}';dest.mkdir(parents=True,exist_ok=True)
 write_rows(dest/'metrics.csv',[row]);write_rows(dest/'rounds.csv',trace);write_rows(dest/'decisions.csv',decisions)
 return row
if __name__=='__main__':
 import sys,concurrent.futures,multiprocessing as mp
 stage=sys.argv[1];p=json.loads((O/('confirmation_plan.json' if stage=='confirmation' else 'plan.json')).read_text())
 if stage=='development':
  jobs=[(pol,s,'nominal',40,h,period,stage) for h in p['planning_horizons'] for period in p['periods'] for pol in ['OptimizedPeriod'] for s in p['development_seeds']]
 else:
  selected=json.loads((O/'selected.json').read_text());h=selected['H'];period=selected['period']
  jobs=[(pol,s,cond,b,h,period,stage) for pol in p['policies'] for cond in p['conditions'] for s in range(p['test_seeds_start'],p['test_seeds_start']+p['test_seed_count']) for b in p['budgets_joules']]
 done=[];begin=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
  for n,row in enumerate(pool.map(job,jobs),1):
   done.append(row)
   if n%20==0:print(stage,n,'/',len(jobs),'seconds',round(time.time()-begin),flush=True)
 write_rows(O/(stage+'_metrics.csv'),done)
 if stage=='development':
  import pandas as pd
  tab=pd.DataFrame(done).groupby(['H','period']).delivered.mean().reset_index().sort_values(['delivered','H','period'],ascending=[False,True,True])
  selected=dict(H=int(tab.iloc[0].H),period=int(tab.iloc[0].period),rule=p['development_selection'],scores=tab.to_dict('records'))
  (O/'selected.json').write_text(json.dumps(selected,indent=2));print(selected,flush=True)
 print('COMPLETE',stage,len(done),flush=True)
