import json,concurrent.futures,multiprocessing as mp
from extension_protocol import O,job
from run_study import write_rows
if __name__=='__main__':
 p=json.loads((O/'confirmation_plan.json').read_text());jobs=[('NoExhaustionNever',s,cond,b,20,20,'confirmation_extra') for cond in p['conditions'] for s in range(9000,9040) for b in p['budgets_joules']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:rows=list(pool.map(job,jobs))
 write_rows(O/'confirmation_extra_metrics.csv',rows);print('DONE',len(rows))
