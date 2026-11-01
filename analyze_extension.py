import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'extension';P=json.loads((O/'plan.json').read_text())
def holm(vals):
 vals=np.asarray(vals);idx=np.argsort(vals);out=np.empty(len(vals));out[idx]=np.minimum(1,np.maximum.accumulate((len(vals)-np.arange(len(vals)))*vals[idx]));return out
def estimate(a,b,seed=94401):
 x=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(seed);boot=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
 raw=1. if np.all(x==0) else float(stats.ttest_1samp(x,0).pvalue)
 return dict(mean_a=float(np.mean(a)),mean_b=float(np.mean(b)),difference=float(x.mean()),relative_pct=float(100*x.mean()/np.mean(b)),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),raw_p=raw,n=len(x))
def main():
 d=pd.read_csv(O/'test_metrics.csv');assert len(d)==2880;assert not d.duplicated(['policy','condition','scenario','budget']).any();assert d.budget_reached.eq(1).all();assert np.allclose(d.spent,d.budget,atol=1e-9)
 d.groupby(['condition','budget','policy']).agg(delivered_mean=('delivered','mean'),delivered_sd=('delivered','std'),control_mean=('control_bits','mean'),telemetry_mean=('telemetry_bits','mean'),requests_mean=('requests','mean'),changed_mean=('changed','mean'),rounds_mean=('rounds','mean'),alive_mean=('alive_final','mean'),runtime_mean=('runtime','mean'),reclusters_mean=('reclusters','mean'),telemetry_energy_mean=('telemetry_energy','mean')).reset_index().to_csv(O/'condition_summary.csv',index=False)
 d.groupby(['budget','policy']).agg(delivered_mean=('delivered','mean'),control_mean=('control_bits','mean'),telemetry_mean=('telemetry_bits','mean'),telemetry_energy_mean=('telemetry_energy','mean'),requests_mean=('requests','mean'),changed_mean=('changed','mean'),runtime_mean=('runtime','mean'),rounds_mean=('rounds','mean'),alive_mean=('alive_final','mean')).reset_index().to_csv(O/'pooled_summary.csv',index=False)
 wide=d[d.budget==40].groupby(['scenario','policy']).delivered.mean().unstack();refs=['Never','Period5','OptimizedPeriod','BaseGate','NoRetryGate','NoSurvivalGate'];rows=[]
 for ref in refs:rows.append(dict(comparator=ref,**estimate(wide.OptimizedGate,wide[ref])))
 hp=holm([r['raw_p'] for r in rows])
 for r,p in zip(rows,hp):r['holm_p']=p
 pd.DataFrame(rows).to_csv(O/'primary_contrasts.csv',index=False)
 secondary=[]
 for cond in P['conditions']:
  for budget in P['budgets_joules']:
   w=d[(d.condition==cond)&(d.budget==budget)].pivot(index='scenario',columns='policy',values='delivered')
   for ref in refs:secondary.append(dict(condition=cond,budget=budget,comparator=ref,**estimate(w.OptimizedGate,w[ref])))
 pd.DataFrame(secondary).to_csv(O/'secondary_contrasts_descriptive.csv',index=False)
 # Factorial attribution on development-selected period vs gate; paired interaction.
 w=d[d.budget==40].groupby(['scenario','policy']).delivered.mean().unstack();interaction=(w.OptimizedGate-w.OptimizedPeriod)-(w.BaseGate-w.SelectedPeriod)
 attribution=dict(cluster_at_period=float((w.OptimizedPeriod-w.SelectedPeriod).mean()),cluster_at_gate=float((w.OptimizedGate-w.BaseGate).mean()),gate_at_base=float((w.BaseGate-w.SelectedPeriod).mean()),gate_at_optimized=float((w.OptimizedGate-w.OptimizedPeriod).mean()),interaction=float(interaction.mean()),status='descriptive attribution; not an additional confirmatory family')
 (O/'attribution.json').write_text(json.dumps(attribution,indent=2))
 colors={'Never':'#7f7f7f','Period5':'#c58c30','SelectedPeriod':'#999999','OptimizedPeriod':'#4a8b65','OptimizedGate':'#235f91'}
 plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for policy in colors:
  x=d[d.policy==policy].groupby('budget').delivered.mean();ax[0].plot(x.index,x.values,'o-',label=policy,color=colors[policy])
 ax[0].set(xlabel='Actual dissipated sensor energy (J)',ylabel='Delivered source samples',title='Energy matched service across four conditions');ax[0].legend(fontsize=8)
 for j,cond in enumerate(P['conditions']):
  z=pd.DataFrame(secondary);z=z[(z.budget==40)&(z.comparator=='Period5')&(z.condition==cond)].iloc[0]
  ax[1].errorbar(j,z.difference,yerr=[[z.difference-z.low],[z.high-z.difference]],fmt='o',color='#235f91',capsize=4)
 ax[1].axhline(0,color='gray',ls='--');ax[1].set_xticks(range(4),['Nominal','Poor channel','Far sink','Heterogeneous'],rotation=20);ax[1].set(ylabel='Additional samples relative to Period5',title='40 J condition results with descriptive 95% CIs');fig.tight_layout();fig.savefig(O/'energy_yield.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 z=d[d.budget==40].groupby('policy')[['delivered','control_bits','telemetry_bits','requests','changed']].mean().reindex(['Never','Period5','SelectedPeriod','BaseGate','OptimizedPeriod','OptimizedNever','OptimizedGate','NoRetryGate','NoSurvivalGate'])
 ax[0].barh(z.index,z.delivered,color=['#235f91' if n=='OptimizedGate' else '#7896ad' for n in z.index]);ax[0].set(xlabel='Delivered samples at 40 J',title='All methods and ablations')
 ax[1].barh(z.index,(z.control_bits-z.telemetry_bits)/1000,label='Reconfiguration');ax[1].barh(z.index,z.telemetry_bits/1000,left=(z.control_bits-z.telemetry_bits)/1000,label='Additional polling');ax[1].set(xlabel='Scheduled sensor control bits (thousands)',title='Control cost including telemetry');ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(O/'ablation_cost.png',dpi=220);plt.close(fig)
 out=dict(episodes=len(d),development_episodes=54,primary=rows,attribution=attribution,total_spent=float(d.spent.sum()),plan_sha256=hashlib.sha256((O/'plan.json').read_bytes()).hexdigest(),selected=json.loads((O/'selected.json').read_text()))
 (O/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
