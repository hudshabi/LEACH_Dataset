"""Separately coded NumPy batch simulator from the written specification.
No original simulator imports, inheritance, transition calls or cluster calls.
B independent networks are advanced together; sensor order is explicit.
"""
import numpy as np
class IndependentBatch:
 def __init__(self,seeds,action_scale=1.,overhead_scale=1.):
  self.rng=[np.random.default_rng(int(s)) for s in seeds];self.b=len(seeds);self.ix=np.arange(self.b);self.action_scale=action_scale;self.overhead_scale=overhead_scale
  self.pos=np.stack([r.uniform(0,100,(100,2)) for r in self.rng]);self.e=np.full((self.b,100),.5);self.states=np.stack([r.random((100,2))<.625 for r in self.rng]);self._geometry();self.t=np.zeros(self.b,int);self.age=np.zeros(self.b,int);self.prev=np.zeros(self.b,int);self.prev_pdr=np.zeros(self.b);self.last=np.zeros((self.b,8));self.setup=np.ones(self.b,bool);self.ch=self.select_heads();self.done=np.zeros(self.b,bool)
 def _geometry(self):
  self.dbs=np.linalg.norm(self.pos-np.array([50.,175.]),axis=2);self.dist=np.linalg.norm(self.pos[:,:,None,:]-self.pos[:,None,:,:],axis=3)
 def select_heads(self):
  alive=self.e>1e-12;n=alive.sum(axis=1);k=np.where(n>0,np.ceil(.05*n).astype(int),0);heads=np.full((self.b,5),-1,int);base=self.e/(.5+self.dbs/200);nearest=np.full((self.b,100),np.inf)
  for j in range(5):
   score=base.copy() if j==0 else base*np.minimum(nearest/70,1)
   score=np.where(alive,score,-np.inf)
   for z in range(j):score[self.ix,np.maximum(heads[:,z],0)]=-np.inf
   chosen=np.argmax(score,axis=1);heads[:,j]=np.where(j<k,chosen,-1);ds=self.dist[self.ix,:,chosen];nearest=np.minimum(nearest,ds)
  return heads
 def observation(self,extended=False):
  base=np.column_stack([self.e.sum(axis=1)/50,self.e.mean(axis=1)/.5,(self.e>1e-12).mean(axis=1),(self.ch>=0).sum(axis=1)/100,np.minimum(self.age/50,1),self.prev]);return np.clip(np.concatenate([base,self.last],axis=1) if extended else base,0,10).astype(np.float32)
 def consume(self,node,amount,enabled):
  node=np.asarray(node);node=np.broadcast_to(node,(self.b,));amount=np.broadcast_to(amount,(self.b,));old=self.e[self.ix,node];success=enabled&(old+1e-15>=amount);self.e[self.ix,node]=np.where(enabled,np.maximum(0,old-amount),old);return success
 def step(self,actions):
  actions=np.asarray(actions,int);valid=~self.done;before=self.e.sum(axis=1);live0=self.e>1e-12;generated=live0.sum(axis=1);z=np.stack([r.random((100,2)) for r in self.rng]);u=np.stack([r.random((100,2,3)) for r in self.rng]);self.states=np.where(self.states,z>=.15,z<.25)
  safe=np.maximum(self.ch,0);forced=(self.ch[:,0]<0)|np.any((self.ch>=0)&(self.e[self.ix[:,None],safe]<=1e-12),axis=1);reform=(actions.astype(bool)|forced|self.setup)&valid
  new=self.select_heads();self.ch=np.where(reform[:,None],new,self.ch);safe=np.maximum(self.ch,0);overhead=np.where(reform,200*generated,0)
  for i in range(100):self.consume(i,1e-5+1e-8*self.dbs[:,i]**2,reform&live0[:,i])
  self.age=np.where(reform,0,self.age+valid);self.prev=np.where(valid,actions,self.prev);self.setup=np.where(valid,False,self.setup)
  received=np.where((self.ch>=0)&(self.e[self.ix[:,None],safe]>0),1,0);attempts=np.zeros(self.b,int);retries=np.zeros(self.b,int);successes=np.zeros(self.b,int);dtotal=np.zeros(self.b);dcount=np.zeros(self.b,int)
  for i in range(100):
   member=live0[:,i]&(self.e[:,i]>1e-12)&~np.any(self.ch==i,axis=1)&(self.ch[:,0]>=0)&valid
   distances=self.dist[self.ix[:,None],i,safe];distances=np.where(self.ch>=0,distances,np.inf);slot=np.argmin(distances,axis=1);c=safe[self.ix,slot];d=distances[self.ix,slot];dtotal+=np.where(member,d,0);dcount+=member
   active=member&(d<=70)&(self.e[self.ix,c]>1e-12);tx=.0002+4e-7*d*d;p=(.1+.8*np.exp(-2.5*d/200))*np.where(self.states[:,i,0],1.,.3)
   for a in range(3):
    sent=self.consume(i,tx,active);rx=self.consume(c,.0002,sent);attempts+=sent;retries+=sent*int(a>0);ok=sent&rx&(u[:,i,0,a]<p);received[self.ix,slot]+=ok;successes+=ok;active=sent&~ok&(self.e[self.ix,c]>1e-12)
  delivered=np.zeros(self.b,int)
  for j in range(5):
   c=safe[:,j];count=received[:,j];active=(self.ch[:,j]>=0)&(count>0)&(self.e[self.ix,c]>1e-12)&valid;d=self.dbs[self.ix,c];tx=.0002+4e-7*d*d;p=(.1+.8*np.exp(-2.5*d/200))*np.where(self.states[self.ix,c,1],1.,.3)
   for a in range(3):
    sent=self.consume(c,tx,active);attempts+=sent;retries+=sent*int(a>0);ok=sent&(u[self.ix,c,1,a]<p);delivered+=np.where(ok,count,0);successes+=ok;active=sent&~ok
  self.t+=valid;energy=before-self.e.sum(axis=1);alive=(self.e>1e-12).sum(axis=1);pdr=delivered/np.maximum(1,generated);variation=np.abs(pdr-self.prev_pdr);self.prev_pdr=pdr;mean_d=np.divide(dtotal,dcount,out=np.zeros_like(dtotal),where=dcount>0);self.last=np.column_stack([pdr,energy/.1,overhead/20000,retries/300,successes/np.maximum(1,attempts),np.ones(self.b),mean_d/200,variation])
  reward=alive/100-.3*energy/.1-self.overhead_scale*.2*overhead/20000-.1*self.e.std(axis=1)/.5-self.action_scale*.05*actions*generated/100;reward=np.where(valid,reward,0)
  self.done|=(alive==0)|(self.t>=1500)
  return dict(round=self.t.copy(),alive=alive,generated=generated,delivered=delivered,pdr=pdr,energy=energy,overhead=overhead,attempts=attempts,retries=retries,recluster=reform.astype(int),forced=forced.astype(int),action=actions,reward=reward)
 @classmethod
 def from_snapshot(cls,snapshot,seeds,action_scale=1,overhead_scale=1):
  obj=cls(seeds,action_scale,overhead_scale);b=obj.b
  for name in ['pos','e','states','last']:
   val=np.asarray(snapshot[name]);setattr(obj,name,np.broadcast_to(val,(b,)+val.shape).copy())
  obj._geometry();ch=np.full(5,-1,int);ch[:len(snapshot['ch'])]=snapshot['ch'];obj.ch=np.broadcast_to(ch,(b,5)).copy()
  for name in ['t','age','prev','prev_pdr','setup']:setattr(obj,name,np.full(b,snapshot[name]))
  obj.done=np.zeros(b,bool);obj.rng=[np.random.default_rng(int(s)) for s in seeds];return obj
