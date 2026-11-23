from pathlib import Path
R=Path(__file__).resolve().parents[2]
p=R/'tests/unit/test_anchor_opd.py'
s=p.read_text()
a='''        with self.assertRaisesRegex(ValueError, "complete"):
            wrapper(**values, opd_scorer=scorer)
'''
b='''        wrapper.zero_grad(set_to_none=True)
        empty = wrapper(**values, opd_scorer=scorer)
        self.assertEqual(empty.loss.item(), 0)
        self.assertEqual(empty.loss_terms.denominator.item(), 0)
        empty.loss.backward()
        self.assertTrue(all(
            p.grad is None or not p.grad.count_nonzero()
            for p in wrapper.draft_model.parameters()
        ))
'''
assert a in s
p.write_text(s.replace(a,b))
p=R/'tests/deferred_objectives_20261122/run_checks.py'
s=(R/'tests/linear_ownership_20261122/run_checks.py').read_text()
a=s.index('files=');b=s.index('\nwith ',a)
files=['test_deferred_objectives.py','test_dpal_renyi.py','test_dpal_local_actors.py','test_dpal_distributed.py','test_vocab_reuse.py','test_composable_eagle3.py','test_anchor_opd.py','test_e2e_tv_causal.py','test_recipe_catalog.py','test_launchers.py']
s=s[:a]+'files='+repr(files)+s[b:]
p.write_text(s.replace('tests/linear_ownership_20261122/tmp','tests/deferred_objectives_20261122/gpu_tmp').replace("work/'pytest.log'","work/'pytest_final.log'"))
