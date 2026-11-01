"""Two-hop research prototype. Not a reproduction of published LEACH-RLC.
Energy-aware deterministic CH heuristic, not MILP. Reliable zero-cost ACK abstraction.
100 nodes, 100x100 m; node-CH hard range 70 m; CH-BS no hard cutoff.
Rayleigh fading is not claimed. Independent source-role Gilbert-Elliott processes
change once per round, fixed over <=3 conditionally independent attempts.
"""
import os,json,csv,time,hashlib
from pathlib import Path
os.environ['MPLBACKEND']='Agg';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
ROOT=Path(__file__).resolve().parent
class TwoHopEnv(gym.Env):
 metadata={'render_modes':[]}
 def __init__(self,link_aware=True,horizon=1500,force_p=None):
  super().__init__();self.link_aware=link_aware;self.horizon=horizon;self.force_p=force_p;self.action_space=spaces.Discrete(2);self.observation_space=spaces.Box(0,10,shape=(14 if link_aware else 6,),dtype=np.float32)
 def reset(self,seed=None,options=None):
  super().reset(seed=seed);self.t=0;self.e=np.full(100,.5);self.pos=self.np_random.uniform(0,100,(100,2));self.dbs=np.linalg.norm(self.pos-[50,175],axis=1);self.dist=np.linalg.norm(self.pos[:,None,:]-self.pos[None,:,:],axis=2);self.states=self.np_random.random((100,2))<.625;self.age=0;self.prev=0;self.last=np.zeros(8);self.prev_pdr=0;self.setup=True;self.ch=self.cluster();self.totals=dict(generated=0,delivered=0,attempts=0,retries=0,energy=0.,overhead=0.,reclusters=0,forced=0);return self.obs(),{}
 def cluster(self):
  alive=np.flatnonzero(self.e>1e-12);k=max(1,int(np.ceil(len(alive)*.05)));chosen=[]
  if not len(alive):return np.array([],dtype=int)
  score=self.e[alive]/(.5+self.dbs[alive]/200)
  for j in range(k):
   merit=score.copy()
   if chosen:merit*=np.minimum(self.dist[alive][:,chosen].min(axis=1)/70,1)
   merit[np.isin(alive,chosen)]=-1;chosen.append(int(alive[np.argmax(merit)]))
  return np.array(chosen)
 def obs(self):
  base=[self.e.sum()/50,self.e.mean()/.5,np.mean(self.e>1e-12),len(self.ch)/100,min(self.age/50,1),self.prev]
  return np.clip(np.array(base+(list(self.last) if self.link_aware else []),dtype=np.float32),0,10)
 def consume(self,i,amount):
  if self.e[i]+1e-15<amount:self.e[i]=0.;return False
  self.e[i]=max(0.,self.e[i]-amount);return True
 def prob(self,d,state):
  if self.force_p is not None:return self.force_p
  return (.1+.8*np.exp(-2.5*d/200))*(1 if state else .3)
 def step(self,action):
  action=int(action);before=self.e.sum();alive0=np.flatnonzero(self.e>1e-12);generated=len(alive0);oldalive=generated
  # Draw all exogenous channel variables before decisions; policy draws cannot shift them.
  z=self.np_random.random((100,2));self.states=np.where(self.states,z>=.15,z<.25);uniforms=self.np_random.random((100,2,3))
  forced=not len(self.ch) or np.any(self.e[self.ch]<=1e-12);reform=bool(action or forced or self.setup);overhead=0
  if reform:
   self.ch=self.cluster();self.age=0;overhead=2*len(alive0)*100
   for i in alive0:self.consume(i,50e-9*100+100e-12*100*self.dbs[i]**2+50e-9*100)
  else:self.age+=1
  self.setup=False;self.prev=action;received={int(i):int(self.e[i]>0) for i in self.ch};attempts=0;retries=0;successes=0;link_dist=[];chset=set(self.ch.tolist())
  for i in alive0:
   if i in chset or self.e[i]<=1e-12:continue
   if not len(self.ch):continue
   ds=self.dist[i,self.ch];c=int(self.ch[np.argmin(ds)]);d=float(ds.min());link_dist.append(d)
   if d>70 or self.e[c]<=1e-12:continue
   tx=.0002+4e-7*d*d
   for a in range(3):
    # Both listening receiver and sender pay for every attempted data frame.
    if not self.consume(i,tx):break
    rxok=self.consume(c,.0002);attempts+=1;retries+=int(a>0)
    if rxok and uniforms[i,0,a]<self.prob(d,self.states[i,0]):received[c]+=1;successes+=1;break
    if self.e[c]<=1e-12:break
  delivered=0
  for c,count in received.items():
   if count==0 or self.e[c]<=1e-12:continue
   tx=.0002+4e-7*self.dbs[c]**2
   for a in range(3):
    if not self.consume(c,tx):break
    attempts+=1;retries+=int(a>0)
    if uniforms[c,1,a]<self.prob(self.dbs[c],self.states[c,1]):delivered+=count;successes+=1;break
  self.t+=1;energy=float(before-self.e.sum());pdr=delivered/max(1,generated);alive=int(np.sum(self.e>1e-12));variation=abs(pdr-self.prev_pdr);self.prev_pdr=pdr
  self.last=np.array([pdr,energy/.1,overhead/20000,retries/300,successes/max(1,attempts),1.,np.mean(link_dist)/200 if link_dist else 0.,variation])
  reward=alive/100-.3*energy/.1-.2*overhead/20000-.1*np.std(self.e)/.5-.05*action
  if self.link_aware:reward+=1.2*pdr-.4*retries/300
  for key,val in dict(generated=generated,delivered=delivered,attempts=attempts,retries=retries,energy=energy,overhead=overhead,reclusters=int(reform),forced=int(forced and not self.setup and not action)).items():self.totals[key]+=val
  info=dict(round=self.t,alive=alive,generated=generated,delivered=delivered,pdr=pdr,energy=energy,overhead=overhead,attempts=attempts,retries=retries,recluster=int(reform),forced=int(forced),action=action)
  return self.obs(),float(reward),alive==0,bool(self.t>=self.horizon and alive>0),info
class Progress(BaseCallback):
 def __init__(self,label):super().__init__();self.label=label;self.start=time.time();self.rows=[]
 def _on_step(self):
  if self.num_timesteps%5000==0:print(self.label,'TRAIN',self.num_timesteps,'sec',round(time.time()-self.start),flush=True)
  for inf in self.locals.get('infos',[]):
   if 'episode' in inf:self.rows.append(dict(step=self.num_timesteps,**inf['episode']))
  return True

def run():
 os.chdir(ROOT);torch.set_num_threads(1);checks={}
 for link in [False,True]:check_env(TwoHopEnv(link),warn=True)
 # deterministic, finite and conservative energy accounting
 a=TwoHopEnv();b=TwoHopEnv();a.reset(seed=72);b.reset(seed=72)
 for t in range(40):
  oa,ra,ta,xa,ia=a.step(t%10==0);ob,rb,tb,xb,ib=b.step(t%10==0);assert np.array_equal(oa,ob) and ia==ib
  assert 0<=ia['delivered']<=ia['generated'];assert a.e.min()>=0
 assert abs(a.totals['energy']+a.e.sum()-50)<1e-8
 checks['deterministic_40_rounds']=True;checks['energy_conservation']=True
 f=TwoHopEnv(force_p=0);f.reset(seed=1)
 for t in range(20):f.step(0)
 assert f.totals['delivered']==0;checks['zero_success_means_no_delivery']=True
 f=TwoHopEnv(force_p=1);f.reset(seed=1);f.step(0);assert f.totals['retries']==0;checks['perfect_channel_needs_no_retry']=True
 f=TwoHopEnv();f.reset(seed=1);f.e[:]=0;f.ch=np.array([],int);o=f.obs();assert np.isfinite(o).all();checks['empty_network_observation_finite']=True
 f=TwoHopEnv(force_p=1);g=TwoHopEnv(force_p=0);f.reset(seed=1);g.reset(seed=1);f.step(0);g.step(0);checks['perfect_round_energy']=f.totals['energy'];checks['failed_round_energy']=g.totals['energy'];assert g.totals['energy']>f.totals['energy'];checks['failures_cost_extra_energy']=True
 Path('corrected_checks.json').write_text(json.dumps(checks,indent=2));print('CORRECTED CHECKS',checks,flush=True)
 models={}
 for name,aware in [('EnergyDQN',False),('LinkDQN',True)]:
  cb=Progress(name);model=DQN('MlpPolicy',TwoHopEnv(aware),learning_rate=1e-3,buffer_size=10000,learning_starts=1000,batch_size=64,tau=.005,gamma=.95,train_freq=4,gradient_steps=1,target_update_interval=500,exploration_fraction=.2,exploration_initial_eps=1,exploration_final_eps=.05,max_grad_norm=10,seed=42,policy_kwargs=dict(net_arch=[128,128,64]),verbose=0,device='cpu')
  model.learn(total_timesteps=50000,callback=cb);model.save(name);Path(name+'_training.json').write_text(json.dumps(cb.rows));models[name]=model
 rows=[];traces=[]
 for name in ['EnergyDQN','LinkDQN','Never','Every20']:
  for seed in range(2000,2020):
   env=TwoHopEnv(name!='EnergyDQN');obs,_=env.reset(seed=seed);fnd=None;ret=0;decisions=0
   while True:
    act=int(models[name].predict(obs,deterministic=True)[0]) if name in models else (0 if name=='Never' else int(env.t%20==0));obs,r,term,trunc,info=env.step(act);ret+=r;decisions+=act
    if fnd is None and info['alive']<100:fnd=env.t
    traces.append(dict(policy=name,seed=seed,**info))
    if term or trunc:break
   z=env.totals;rows.append(dict(policy=name,seed=seed,rounds=env.t,FND=fnd if fnd else env.t,FND_censored=fnd is None,LND=env.t if term else '',LND_censored=not term,alive_final=int((env.e>0).sum()),PDR=z['delivered']/max(1,z['generated']),packets_generated=z['generated'],packets_delivered=z['delivered'],packets_lost=z['generated']-z['delivered'],energy_total=z['energy'],energy_per_round=z['energy']/env.t,energy_per_delivered=z['energy']/z['delivered'] if z['delivered'] else '',overhead_bits_per_round=z['overhead']/env.t,retries=z['retries'],recluster_count=z['reclusters'],requested_recluster_count=decisions,reward=ret))
   print('CORRECTED EVAL',name,seed,env.t,round(rows[-1]['PDR'],4),flush=True)
  with open('corrected_per_seed.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with open('corrected_per_round.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=traces[0]);w.writeheader();w.writerows(traces)
 Path('corrected_manifest.json').write_text(json.dumps(dict(training_steps_per_model=50000,training_seed=42,evaluation_seeds=list(range(2000,2020)),clustering='energy-distance spatial heuristic; no MILP',ack='ideal and zero energy',aggregation='one 4000-bit CH frame represents all received samples plus CH sample',control='100-bit status to BS plus 100-bit configuration reception per alive node; BS energy not charged',range='70 m member-CH; no hard CH-BS range',channel='GE per source-role, updated once per round, held fixed over up to three attempts; initial stationary good probability .625',seed_scope='common topology and pre-drawn source-role channel uniforms across policies',reward_baseline='alive fraction, energy, overhead, dispersion and decision costs',reward_link='baseline plus delivery reward and retry cost',initial_energy=50,model_limits='no MILP; no CSMA/interference/delay/ACK energy; one training seed; combined state+reward change not separately ablated'),indent=2))
 print('CORRECTED COMPLETE',flush=True)
if __name__=='__main__':run()
