from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2]
def edit(name,fn):
 p=R/name;s=fn(p.read_text());ast.parse(s);p.write_text(s)
# Keep rank-local compiler caches rooted in the same checkout after adding a directory level.
edit('legospec/training/runtime/runner.py',lambda s:s.replace('Path(__file__).resolve().parents[2]','Path(__file__).resolve().parents[3]'))
# Patch the actual owner now that setup no longer re-exports feature helpers.
for name in ['test_target_feature_contract.py','test_hidden_feature_cache.py','test_sglang_gemma4_training.py']:
 def fix(s):
  s=s.replace('patch.object(common,','patch.object(target_features,').replace('patch.object(setup,','patch.object(target_features,')
  s=s.replace('patch.object(\n                setup,','patch.object(\n                target_features,').replace('patch.object(\n        setup,','patch.object(\n        target_features,')
  # AutoTokenizer belongs to features for component binding, even though setup also uses it for target ranks.
  s=s.replace('runtime_setup.AutoTokenizer','target_features.AutoTokenizer')
  return s
 edit('tests/unit/'+name,fix)
def logging_tests(s):
 a=s.index('    import legospec.training.runtime.setup as training')
 b=s.index('\n    class Tracker:',a)
 s=s[:a]+'''    from legospec.training.runtime import runner
    from legospec.training.execution import step, windows
    from legospec.training.logger import logging

    adapter = Adapter()
    monkeypatch.setattr(runner, "enter_phase", adapter.enter_phase)
    monkeypatch.setattr(step, "compute_step", adapter.compute_step)
    monkeypatch.setattr(logging, "log_metrics", adapter.log_metrics)
    monkeypatch.setattr(logging, "detail_metrics", adapter.detail_metrics)
    monkeypatch.setattr(windows, "prepare_optimizer_window", lambda *a: None)
''' +s[b:]
 return s
edit('tests/unit/test_optimizer_update_logging.py',logging_tests)
def disagg(s):
 s=s.replace('from legospec.training.runtime import setup as training','from legospec.training.execution import windows\nfrom legospec.training.data import batch\nfrom legospec.training.target import streaming')
 s=s.replace('    adapter = training\n','').replace('adapter.prepare_optimizer_window','windows.prepare_optimizer_window').replace('adapter.anchor_layout','batch.anchor_layout').replace('adapter.annotate_target_packet','streaming.annotate_target_packet')
 return s
edit('tests/unit/test_disaggregated_streaming_windows.py',disagg)
edit('tests/unit/test_dpal_renyi.py',lambda s:s.replace('    train.component_normalization = False\n',''))
def packed(s):
 a=s.index('    with torch.no_grad():\n        record = legacy(**data, opd_scorer=scorer, metadata_only=True)')
 b=s.index('    packed = deepcopy(legacy)',a)
 s=s[:a]+'''    from unittest.mock import patch
    from legospec.training.execution import sampled

    records = []
    batch_type = sampled.SampledOPDBatch

    def capture(*args, **kwargs):
        record = batch_type(*args, **kwargs)
        records.append(record)
        return record

    with patch.object(sampled, "SampledOPDBatch", side_effect=capture), torch.no_grad():
        legacy(**data, opd_scorer=scorer)
    record, = records
''' +s[b:]
 return s
edit('tests/unit/test_packed_training.py',packed)
print('Updated runtime-relative path and relocated test mocks; sampled replay test now uses active E2E-TV collection.')
