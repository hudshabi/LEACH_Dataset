"""Fresh simulations under a hard cap on actual dissipated sensor energy."""
import os,json,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,torch
from stable_baselines3 import DQN
from run_study import FactorEnv,write_rows
R=Path(__file__).resolve().parent;O=R/'final_validation';P=json.loads((R/'final_validation_plan.json').read_text())
class EnergyBudgetEnv(FactorEnv):
 def __init__(self,budget,state=False,reward=False):self.energy_cap=float(budget);self.used=0.;super().__init__(state,reward)
 def reset(self,**kwargs):self.used=0.;return super().reset(**kwargs)
 def consume(self,i,amount):
  allowance=max(0.,self.energy_cap-self.used);node=float(self.e[i]);debit=min(node,float(amount),allowance);self.e[i]=max(0.,node-debit);self.used+=debit
  # Completion requires both original node affordability and global allowance.
  return bool(node+1e-15>=amount and allowance+1e-15>=amount)
def job(arg):
 torch.set_num_threads(1);name,seed=arg;model=None
 if name in ['S0R0','S1R1']:model=DQN.load(str(R/'runs'/f'{name}_{seed}'/'model.zip'),device='cpu')
 state=name=='S1R1';rows=[];logs=[]
 for budget in P['energy_budgets_joules']:
  for scenario in P['fresh_scenarios']:
   env=EnergyBudgetEnv(budget,state,state);obs,_=env.reset(seed=scenario);delivered=generated=overhead=0;requests=0
   while env.used<budget-1e-10:
    action=int(model.predict(obs,deterministic=True)[0]) if model is not None else (0 if name=='Never' else int(env.t%int(name[6:])==0));obs,r,t,x,i=env.step(action);delivered+=i['delivered'];generated+=i['generated'];overhead+=i['overhead'];requests+=action
    logs.append(dict(policy=name,train_seed=seed,budget=budget,scenario=scenario,round=env.t,spent=env.used,delivered=i['delivered'],energy=i['energy'],alive=i['alive'],action=action,ch_hash=i['ch_hash']))
    if t or x:break
   assert abs(env.used-budget)<1e-9,(name,seed,budget,scenario,env.used)
   assert abs(env.e.sum()+env.used-50)<1e-8
   rows.append(dict(policy=name,train_seed=seed,budget=budget,scenario=scenario,spent=env.used,delivered=delivered,energy_per_delivered=env.used/delivered if delivered else float('nan'),attempted_rounds=env.t,alive_final=int((env.e>1e-12).sum()),generated=generated,PDR=delivered/max(generated,1),scheduled_control_bits=overhead,requests=requests))
 f=O/f'budget_{name}_{seed}';f.mkdir(exist_ok=True);write_rows(f/'metrics.csv',rows);write_rows(f/'rounds.csv',logs);print('BUDGET DONE',name,seed,flush=True);return rows
if __name__=='__main__':
 import concurrent.futures,multiprocessing as mp
 jobs=[(n,-1) for n in ['Never','Period5','Period20']]+[(n,s) for n in ['S0R0','S1R1'] for s in P['training_seeds']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:rows=[r for group in pool.map(job,jobs) for r in group]
 write_rows(O/'equal_budget_metrics.csv',rows);print('BUDGET EPISODES',len(rows),flush=True)
