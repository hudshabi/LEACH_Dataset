import json
import numpy as np,pandas as pd
from analyze_extension import estimate,holm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent/'extension'
def main():
 d=pd.concat([pd.read_csv(O/'confirmation_metrics.csv'),pd.read_csv(O/'confirmation_extra_metrics.csv')],ignore_index=True);assert len(d)==5120;assert d.budget_reached.eq(1).all();assert np.allclose(d.spent,d.budget,atol=1e-9);assert not d.duplicated(['policy','scenario','condition','budget']).any()
 refs=['Never','Period5','SelectedPeriod','OptimizedPeriod','NoRetryNever','NoSurvivalNever','NoExhaustionNever']
 w=d[d.budget==40].groupby(['scenario','policy']).delivered.mean().unstack();rows=[]
 for ref in refs:rows.append(dict(comparator=ref,**estimate(w.OptimizedNever,w[ref],91401)))
 for r,p in zip(rows,holm([r['raw_p'] for r in rows])):r['holm_p']=p
 pd.DataFrame(rows).to_csv(O/'confirmation_primary.csv',index=False)
 secs=[]
 for cond in d.condition.unique():
  w=d[(d.budget==40)&(d.condition==cond)].pivot(index='scenario',columns='policy',values='delivered')
  for ref in refs:secs.append(dict(condition=cond,comparator=ref,**estimate(w.OptimizedNever,w[ref],91401)))
 for r,p in zip(secs,holm([r['raw_p'] for r in secs])):r['holm_p']=p
 pd.DataFrame(secs).to_csv(O/'confirmation_condition_contrasts.csv',index=False)
 desc=[]
 for b in [10,20,30,40]:
  w=d[d.budget==b].groupby(['scenario','policy']).delivered.mean().unstack()
  for ref in refs:desc.append(dict(budget=b,comparator=ref,**estimate(w.OptimizedNever,w[ref],91401)))
 pd.DataFrame(desc).to_csv(O/'confirmation_budget_descriptive.csv',index=False)
 for by,name in [(['budget','policy'],'confirmation_summary'),(['condition','budget','policy'],'confirmation_condition_summary')]:
  z=d.groupby(by).agg(delivered=('delivered','mean'),SD=('delivered','std'),control_bits=('control_bits','mean'),reclusters=('reclusters','mean'),rounds=('rounds','mean'),alive=('alive_final','mean'),runtime=('runtime','mean'),requests=('requests','mean'),changed=('changed','mean'),spent=('spent','mean')).reset_index();z.to_csv(O/(name+'.csv'),index=False)
 (O/'confirmation_analysis.json').write_text(json.dumps(dict(episodes=len(d),primary=rows,secondary=secs),indent=2))
 plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.spines.right':False,'axes.spines.top':False})
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for name,color in [('Never','#888888'),('Period5','#b6812b'),('SelectedPeriod','#65735d'),('OptimizedPeriod','#5e9c76'),('OptimizedNever','#235f91')]:
  s=d[d.policy==name].groupby('budget').delivered.mean();ax[0].plot(s.index,s.values,'o-',label=name,color=color)
 ax[0].legend(fontsize=7);ax[0].set(xlabel='Actual sensor energy spent (J)',ylabel='Delivered source samples',title='Fresh confirmation cohort')
 z=pd.DataFrame(secs);z=z[z.comparator=='Never']
 for j,(_,r) in enumerate(z.iterrows()):ax[1].errorbar(j,r.difference,yerr=[[r.difference-r.low],[r.high-r.difference]],fmt='o',capsize=4,color='#235f91')
 ax[1].axhline(0,color='gray',ls='--');ax[1].set_xticks(range(4),['Nominal','Poor channel','Far sink','Heterogeneous'],rotation=20);ax[1].set(ylabel='Additional samples at 40 J versus Never',title='Condition estimates and descriptive 95% CIs');fig.tight_layout();fig.savefig(O/'confirmation_yield.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4));order=['Never','Period5','SelectedPeriod','OptimizedPeriod','OptimizedNever','NoRetryNever','NoSurvivalNever','NoExhaustionNever'];z=d[d.budget==40].groupby('policy').mean(numeric_only=True).reindex(order)
 ax[0].barh(order,z.delivered,color=['#235f91' if n=='OptimizedNever' else '#8da8b9' for n in order]);ax[0].set(xlabel='Delivered samples at 40 J',title='Formation and timing ablations')
 ax[1].barh(order,z.control_bits/1000,color=['#235f91' if n=='OptimizedNever' else '#8da8b9' for n in order]);ax[1].set(xlabel='Scheduled control bits (thousands)',title='Communication cost at matched energy');fig.tight_layout();fig.savefig(O/'confirmation_ablation.png',dpi=220);plt.close(fig)
 print(pd.DataFrame(rows).to_string(index=False));print(z[['delivered','control_bits','reclusters','rounds','alive_final','runtime']].to_string())
if __name__=='__main__':main()
