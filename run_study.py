"""Reproducible 2x2 state/reward experiment. No MILP or guaranteed favorable outcome.
Original radio/channel dynamics are preserved in base_simulator.py.
Policy selection uses development seeds only. Test seeds and all outcomes are retained.
"""
import os,json,csv,time,hashlib,sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import numpy as np
from base_simulator import TwoHopEnv
ROOT=Path(__file__).resolve().parent;DESIGN=json.loads((ROOT/'design.json').read_text())
class FactorEnv(TwoHopEnv):
 def __init__(self,state=False,reward=False,normalize=True,**kw):
  self.reward_aware=reward;self.normalize=normalize;super().__init__(link_aware=state,**kw)
 def step(self,action):
  alive_before=int(np.sum(self.e>1e-12));obs,_,term,trunc,info=super().step(action)
  terms=dict(alive_component=info['alive']/100,energy_component=-.3*info['energy']/.1,overhead_component=-.2*info['overhead']/20000,dispersion_component=-.1*np.std(self.e)/.5,action_component=-.05*int(action)*(alive_before/100 if self.normalize else 1),delivery_component=1.2*info['pdr'] if self.reward_aware else 0.,retry_component=-.4*info['retries']/300 if self.reward_aware else 0.)
  rew=float(sum(terms.values()));info.update(terms);info['reward']=rew;info['alive_before']=alive_before;info['ch_set']=','.join(map(str,self.ch.tolist()))
  info['ch_hash']=hashlib.sha256(info['ch_set'].encode()).hexdigest()[:16]
  return obs,rew,term,trunc,info

def rollout(policy,scenario,model=None,state=False,reward=False,train_seed=-1,folder=None,trace=True,normalize=True):
 e=FactorEnv(state,reward,normalize);obs,_=e.reset(seed=scenario);anchor=100;T10=T50=FND=None;decisions=0;ret=0;tr=[]
 while True:
  if model is not None:action=int(model.predict(obs,deterministic=True)[0])
  elif policy=='Never':action=0
  elif policy.startswith('Period'):action=int(e.t%int(policy[6:])==0)
  elif policy.startswith('Drop'):action=int(np.sum(e.e>1e-12)<=anchor*(1-float(policy[4:])/100))
  else:raise ValueError(policy)
  ch_pre=','.join(map(str,e.ch.tolist()));obs,r,term,trunc,info=e.step(action);decisions+=action;ret+=r
  if info['recluster']:anchor=info['alive']
  if FND is None and info['alive']<100:FND=e.t
  if T10 is None and info['alive']<=90:T10=e.t
  if T50 is None and info['alive']<=50:T50=e.t
  if trace:
   line=dict(policy=policy,train_seed=train_seed,scenario=scenario,**info)
   if scenario==4000:line['ch_before']=ch_pre
   else:line.pop('ch_set');line['ch_before']=''
   tr.append(line)
  if term or trunc:break
 t=e.totals
 first=tr[:200]
 # The trace is retained for every scored episode, so C200 is computed directly.
 coverage=sum(z['delivered'] for z in first)/20000 if trace else None
 row=dict(policy=policy,train_seed=train_seed,scenario=scenario,FND=FND if FND is not None else e.t,T10=T10 if T10 is not None else e.t,T50=T50 if T50 is not None else e.t,LND=e.t,FND_censored=FND is None,T10_censored=T10 is None,T50_censored=T50 is None,LND_censored=not term,PDR=t['delivered']/max(1,t['generated']),C200=coverage,energy_per_round=t['energy']/e.t,energy_per_delivered=t['energy']/max(1,t['delivered']),overhead_bits_per_round=t['overhead']/e.t,generated=t['generated'],delivered=t['delivered'],energy_total=t['energy'],requested_actions=decisions,executed_reclusters=t['reclusters'],return_total=ret)
 return row,tr

def write_rows(path,rows):
 if rows:
  fields=list(dict.fromkeys(k for r in rows for k in r))
  with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def train_job(args):
 config,seed=args;import torch
 from stable_baselines3 import DQN
 from stable_baselines3.common.callbacks import BaseCallback
 torch.set_num_threads(1);state=config[1]=='1';reward=config[3]=='1';normalize=not config.endswith('_U');folder=ROOT/'runs'/f'{config}_{seed}';folder.mkdir(parents=True,exist_ok=True);start=time.time()
 class Progress(BaseCallback):
  def __init__(self):super().__init__();self.episodes=[]
  def _on_step(self):
   if self.num_timesteps%10000==0:
    (folder/'progress.json').write_text(json.dumps(dict(config=config,seed=seed,steps=self.num_timesteps,elapsed=time.time()-start)))
   for info in self.locals.get('infos',[]):
    if 'episode' in info:self.episodes.append(dict(step=self.num_timesteps,**info['episode']))
   return True
 cb=Progress();model=DQN('MlpPolicy',FactorEnv(state,reward,normalize=normalize),learning_rate=.001,buffer_size=10000,learning_starts=1000,batch_size=64,tau=.005,gamma=.95,train_freq=4,gradient_steps=1,target_update_interval=500,exploration_fraction=.2,exploration_initial_eps=1.,exploration_final_eps=.05,max_grad_norm=10,seed=seed,policy_kwargs=dict(net_arch=[128,128,64]),verbose=0,device='cpu')
 model.learn(total_timesteps=DESIGN['train_steps'],callback=cb);model.save(str(folder/'model'));write_rows(folder/'training_episodes.csv',cb.episodes)
 rows=[];logs=[]
 for scenario in DESIGN['test_seeds']:
  row,tr=rollout(config,scenario,model,state,reward,seed,normalize=normalize);rows.append(row);logs.extend(tr)
 write_rows(folder/'metrics.csv',rows);write_rows(folder/'round_log.csv',logs)
 (folder/'complete.json').write_text(json.dumps(dict(config=config,seed=seed,steps=DESIGN['train_steps'],episodes=len(rows),elapsed_seconds=time.time()-start)))
 return dict(config=config,seed=seed,seconds=round(time.time()-start))

def controls():
 policies=['Never']+['Period'+str(p) for p in DESIGN['periods']]+['Drop'+str(int(q*100)) for q in DESIGN['alive_drop_thresholds']]
 dev=[];devlogs=[]
 for policy in policies:
  for scenario in DESIGN['development_seeds']:
   row,tr=rollout(policy,scenario);dev.append(row)
 write_rows(ROOT/'development_controls.csv',dev)
 import pandas as pd
 z=pd.DataFrame(dev).groupby('policy').agg(C200=('C200','mean'),cost=('energy_per_delivered','mean')).sort_values(['C200','cost'],ascending=[False,True]);selected=z.index[0]
 (ROOT/'selected_comparator.json').write_text(json.dumps(dict(policy=selected,rule=DESIGN['selection_rule'],development_summary=z.reset_index().to_dict('records')),indent=2))
 rows=[];logs=[]
 for policy in policies:
  for scenario in DESIGN['test_seeds']:
   row,tr=rollout(policy,scenario);rows.append(row);logs.extend(tr)
 write_rows(ROOT/'test_controls.csv',rows);write_rows(ROOT/'control_round_log.csv',logs)
 print('CONTROL SELECTION',selected,flush=True)

def mechanical_checks():
 from stable_baselines3.common.env_checker import check_env
 for state in [0,1]:
  for reward in [0,1]:check_env(FactorEnv(state,reward),warn=True)
 a=FactorEnv(1,1);b=FactorEnv(1,1);a.reset(seed=7);b.reset(seed=7)
 for i in range(100):
  oa,ra,ta,xa,ia=a.step(i%20==0);ob,rb,tb,xb,ib=b.step(i%20==0)
  assert np.array_equal(oa,ob) and ia==ib and ra==rb
  assert abs(sum(ia[k] for k in ia if k.endswith('_component'))-ra)<1e-10
  assert abs(a.e.sum()+a.totals['energy']-50)<1e-8
 # Two observations and two rewards must not change the physical transition for identical actions.
 a=FactorEnv(0,0);b=FactorEnv(1,1);a.reset(seed=9);b.reset(seed=9)
 for i in range(100):
  _,_,_,_,ia=a.step(i%12==0);_,_,_,_,ib=b.step(i%12==0)
  for key in ['alive','generated','delivered','energy','overhead','ch_hash']:assert ia[key]==ib[key]
 (ROOT/'checks.json').write_text(json.dumps(dict(repeated_seed_trajectory=True,reward_components_sum=True,energy_conservation=True,factor_physics_identity=True,gym_checks=4),indent=2))
if __name__=='__main__':
 import concurrent.futures,multiprocessing as mp
 mechanical_checks()
 if '--checks-only' in sys.argv:print('CHECKS PASSED');sys.exit(0)
 jobs=[(c,s) for c in DESIGN['factorial'] for s in DESIGN['train_seeds']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
  futures=[pool.submit(train_job,j) for j in jobs]+[pool.submit(controls)]
  for f in concurrent.futures.as_completed(futures):print('FINISHED',f.result(),flush=True)
 import torch,gymnasium,stable_baselines3,scipy,platform
 (ROOT/'versions.json').write_text(json.dumps(dict(torch=torch.__version__,gymnasium=gymnasium.__version__,sb3=stable_baselines3.__version__,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version()),indent=2));print('ALL COMPLETE',flush=True)
