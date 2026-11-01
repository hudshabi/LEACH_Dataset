import unittest
import numpy as np
from extension_protocol import ExtensionEnv,episode
class ExtensionTests(unittest.TestCase):
 def test_retry_expectations(self):
  e=ExtensionEnv();s,a=e.moments(np.array([0.,1.]));self.assertAlmostEqual(s[0],0);self.assertAlmostEqual(a[0],3);self.assertAlmostEqual(s[1],.625+.375*(1-.7**3));self.assertTrue(a[1]>1)
 def test_no_channel_genie(self):
  a=ExtensionEnv(10,optimized=True);a.reset(seed=999);x=a.cluster();v=a.forecast(x,True);a.states=~a.states;self.assertTrue(np.array_equal(x,a.cluster()));self.assertTrue(np.array_equal(v,a.forecast(x,True)))
 def test_swap_not_worse_than_initializer(self):
  from base_simulator import TwoHopEnv
  for cond in ['nominal','poor_channel','far_sink','heterogeneous']:
   e=ExtensionEnv(40,cond,True);e.reset(seed=996);base=TwoHopEnv.cluster(e);opt=e.cluster();self.assertGreaterEqual(e.forecast(opt,True)[0]+1e-10,e.forecast(base,True)[0]);self.assertEqual(len(set(opt)),len(opt));self.assertEqual(len(opt),len(base))
 def test_monitoring_and_equal_caps(self):
  for name in ['Never','Period5','BaseGate','OptimizedGate','NoRetryGate','NoSurvivalGate']:
   r,trace,_=episode(name,998,'nominal',10,trace=True);self.assertAlmostEqual(r['spent'],10,places=9);self.assertGreaterEqual(r['control_bits'],r['telemetry_bits']);self.assertGreater(r['delivered'],0)
   if 'Gate' in name:self.assertGreater(r['telemetry_energy'],0)
 def test_deterministic_and_nonnegative(self):
  a,_,_=episode('OptimizedGate',997,'heterogeneous',10);b,_,_=episode('OptimizedGate',997,'heterogeneous',10)
  for k in a:
   if k!='runtime':self.assertEqual(a[k],b[k],k)
 def test_baseline_original_parity(self):
  from run_equal_budget import EnergyBudgetEnv
  a=ExtensionEnv(10);b=EnergyBudgetEnv(10);a.reset(seed=994);b.reset(seed=994)
  while a.used<10-1e-10:
   aa=a.step(0);bb=b.step(0);self.assertTrue(np.array_equal(a.e,b.e));self.assertEqual(aa[4],bb[4])

class ConfirmationIntegrity(unittest.TestCase):
 def test_complete_confirmation_and_caps(self):
  import pandas as pd
  from extension_protocol import O
  d=pd.concat([pd.read_csv(O/'confirmation_metrics.csv'),pd.read_csv(O/'confirmation_extra_metrics.csv')]);self.assertEqual(len(d),5120);self.assertFalse(d.duplicated(['policy','scenario','condition','budget']).any());self.assertTrue(d.groupby(['policy','condition','budget']).size().eq(40).all());np.testing.assert_allclose(d.spent,d.budget,atol=1e-9);self.assertTrue(d.budget_reached.eq(1).all())
 def test_no_optional_actions_in_selected_policy(self):
  import pandas as pd
  from extension_protocol import O
  d=pd.read_csv(O/'confirmation_metrics.csv');x=d[d.policy=='OptimizedNever'];self.assertTrue(x.requests.eq(0).all());self.assertTrue(x.telemetry_bits.eq(0).all());self.assertTrue(x.telemetry_energy.eq(0).all())
 def test_primary_replication_unit_and_family(self):
  import pandas as pd
  from extension_protocol import O
  d=pd.read_csv(O/'confirmation_primary.csv');self.assertEqual(len(d),7);self.assertTrue(d.n.eq(40).all());self.assertTrue((d.holm_p>=d.raw_p-1e-15).all());self.assertTrue(d[d.comparator=='Never'].holm_p.lt(.05).all());self.assertTrue(d[d.comparator=='Period5'].holm_p.gt(.05).all())

if __name__=='__main__':unittest.main()
