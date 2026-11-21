"""D-PAL math, pinned author equations, reductions, heads and training plumbing."""

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from litespec.core.composable.batch import prepare_layout
from litespec.core.composable.objectives import build_objective
from litespec.core.composable.objectives.base import Supervision
from litespec.core.composable.objectives import dpal_renyi as dpal_module
from litespec.core.composable.objectives.dpal_renyi import renyi_half_and_overlap, dpal_credit
from litespec.core.composable.objectives.selector import LossBranches
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
from litespec.modeling.draft.composable.contracts import DraftOutput
from litespec.modeling.draft.composable.export import export_composable
from litespec.modeling.draft.composable.loss_config import normalize_objective


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests/artifacts/dpal_renyi"
REFERENCE = ROOT / "tests/reference/dpal_upstream/specforge/algorithms/common/dflash_family_model.py"


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(423)
    torch.set_num_threads(1)


def objective(reduction="complete_block_mean", **kwargs):
    return build_objective(dict(type="dpal_renyi", reduction=reduction, **kwargs))


def evaluate(q, p, mask, reduction="complete_block_mean", counts=None, **kwargs):
    return objective(reduction, **kwargs)(DraftOutput(q, draft_logits=q), Supervision(
        labels=torch.zeros_like(mask, dtype=torch.long), prediction_mask=mask,
        target_logits=p, sample_anchor_counts=counts))


def author_functions():
    # Execute only two reviewed tensor functions, with the compile decorator
    # removed. Never import the upstream trainer or its dependencies.
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == 'a558986b4fa05eeb1a46fd531f91eddb7f2dcfeb94c9896c78b0147a3475bf60'
    source = ast.parse(REFERENCE.read_text())
    unary = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_unary_renyi_overlap")
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and any(
        isinstance(f, ast.FunctionDef) and f.name == "_dpace_weight" for f in n.body))
    weight = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_dpace_weight")
    unary.decorator_list = weight.decorator_list = []
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[unary, weight], type_ignores=[]), str(REFERENCE), "exec"), scope)
    return scope[unary.name], scope[weight.name]


@pytest.mark.parametrize("rho", [0., .3, .5, 1.])
def test_pinned_author_actor_credit_and_gradient(rho):
    unary, weight = author_functions()
    q = torch.randn(2, 4, 7, 19, requires_grad=True)
    p = torch.randn_like(q, requires_grad=True)
    mask = (torch.rand(q.shape[:-1]) > .3).float()
    actor, overlap = renyi_half_and_overlap(q, p)
    a, o = unary(q, p)
    torch.testing.assert_close(actor, a, rtol=0, atol=0)
    torch.testing.assert_close(overlap, o, rtol=0, atol=0)
    credit = dpal_credit(overlap, mask, rho)
    with torch.no_grad():
        expected = weight(SimpleNamespace(dpace_alpha=rho), o, mask, mask.bool(), "dpace")
    torch.testing.assert_close(credit, expected, rtol=0, atol=0)
    grad = torch.autograd.grad((actor * credit * mask).sum(), q)[0]
    expected_grad = torch.autograd.grad((a * expected * mask).sum(), q)[0]
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)
    assert not overlap.requires_grad and not credit.requires_grad and p.grad is None


def test_closed_form_gradient_and_finite_difference():
    q = torch.tensor([[.2, -.6, 1.3]], requires_grad=True)
    p = torch.tensor([[.8, -.1, -.4]], requires_grad=True)
    actor, _ = renyi_half_and_overlap(q, p)
    grad = torch.autograd.grad(actor.sum(), q, retain_graph=True)[0]
    r = ((p.detach().log_softmax(-1) + q.detach().log_softmax(-1)) * .5).softmax(-1)
    torch.testing.assert_close(grad, q.detach().softmax(-1) - r, rtol=1e-5, atol=1e-7)
    for i in range(3):
        delta = torch.zeros_like(q); delta[0, i] = .002
        finite = (renyi_half_and_overlap(q.detach() + delta, p)[0]
                  - renyi_half_and_overlap(q.detach() - delta, p)[0]) / .004
        torch.testing.assert_close(grad[0, i], finite.squeeze(), rtol=.002, atol=.0001)
    actor.sum().backward()
    assert p.grad is None


def test_equal_distributions_and_one_hot_ce_dpace_limit():
    q = torch.randn(2, 3, 7, 11, requires_grad=True)
    mask = torch.ones(q.shape[:-1])
    same = evaluate(q, q.detach(), mask)
    assert abs(same.loss.item()) < 4e-6
    assert torch.autograd.grad(same.loss, q)[0].abs().max() < 1e-6
    labels = torch.randint(11, mask.shape)
    p = torch.full_like(q, -torch.inf).scatter(-1, labels[..., None], 0.)
    local, overlap = renyi_half_and_overlap(q, p)
    ce = -q.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    torch.testing.assert_close(local, ce)
    torch.testing.assert_close(overlap, ce.detach().neg().exp())
    got = evaluate(q, p, mask, rho=.3)
    old = build_objective(dict(type="dpace", alpha=.3, loss_decay_gamma=None))(
        DraftOutput(q, draft_logits=q), Supervision(labels, mask))
    torch.testing.assert_close(got.loss, old.loss)
    ga = torch.autograd.grad(got.loss, q)[0]
    gb = torch.autograd.grad(old.loss, q)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_extremes_masked_nan_empty_chunks_and_diagnostics(dtype):
    q = (torch.randn(2, 3, 7, 13) * 1000).to(dtype).requires_grad_()
    p = (torch.randn_like(q) * 1000).requires_grad_()
    mask = torch.ones(q.shape[:-1]); mask[0, 1:, :] = 0; mask[1, 2, -1] = 0
    # Native reduction excludes the entire incomplete block, including NaNs.
    with torch.no_grad():
        q[0, 1:] = torch.nan; p[0, 1:] = torch.nan
        q[1, 2] = torch.nan; p[1, 2] = torch.nan
    out = evaluate(q, p, mask)
    assert torch.isfinite(out.loss) and out.loss.dtype == torch.float32
    assert out.loss_terms.denominator == 3
    for terms in out.ratio_metrics.values():
        assert terms.denominator == 21 and not terms.numerator.requires_grad
    out.loss.backward()
    assert torch.isfinite(q.grad).all() and p.grad is None
    empty = evaluate(q, p, torch.zeros_like(mask))
    assert empty.loss == 0 and empty.loss_terms.denominator == 0
    assert torch.autograd.grad(empty.loss, q)[0].abs().sum() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA loss compilation"))])
def test_masked_vocab_path_matches_eager_loss_metrics_and_gradients(monkeypatch, dtype, device):
    # Keep the vocabulary tail unaligned and exercise both FP32 final logits
    # and BF16 LM-head logits, with BF16 teacher logits in either case.
    q = torch.randn(2, 3, 7, 67, device=device, dtype=dtype)
    p = torch.randn_like(q, dtype=torch.bfloat16)
    mask = torch.ones(q.shape[:-1], device=device)
    mask[0, 1:] = 0
    mask[1, 2, -1] = 0
    q[0, 1:] = torch.nan
    p[1, 2] = torch.nan
    p[0, 0, 1, 1:] = -torch.inf  # A valid one-hot teacher row.
    q.requires_grad_()
    p.requires_grad_()

    monkeypatch.setenv("LITESPEC_LOSS_BACKEND", "torch")
    ref = evaluate(q, p, mask)
    ref_grad = torch.autograd.grad(ref.loss, q)[0]
    monkeypatch.setenv("LITESPEC_LOSS_BACKEND", "inductor")
    if device == "cpu":
        # Exercise the optimized formulas and custom backward on CPU too.
        monkeypatch.setattr(dpal_module, "_should_compile", lambda _: True)
    actual = evaluate(q, p, mask)
    actual_grad, teacher_grad = torch.autograd.grad(actual.loss, (q, p), allow_unused=True)
    torch.testing.assert_close(actual.loss, ref.loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.loss_terms.denominator, ref.loss_terms.denominator)
    for key, terms in ref.ratio_metrics.items():
        torch.testing.assert_close(actual.ratio_metrics[key].numerator, terms.numerator,
                                   rtol=2e-5, atol=2e-6)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual_grad, ref_grad, rtol=8e-3, atol=2e-6)
    else:
        torch.testing.assert_close(actual_grad, ref_grad, rtol=3e-5, atol=2e-7)
    assert teacher_grad is None
    assert not actual_grad[0, 1:].count_nonzero() and not actual_grad[1, 2].count_nonzero()
    empty = evaluate(q, p, torch.zeros_like(mask))
    assert empty.loss == 0 and empty.loss_terms.denominator == 0
    assert not torch.autograd.grad(empty.loss, q)[0].count_nonzero()


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
def test_invalid_active_logits_fail(bad):
    q, p = torch.randn(1, 1, 2, 5), torch.randn(1, 1, 2, 5)
    q[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        evaluate(q, p, torch.ones(1, 1, 2))


def test_sequence_reduction_holes_unequal_anchors_and_prechunk_counts():
    q, p = torch.randn(3, 4, 7, 9), torch.randn(3, 4, 7, 9)
    mask = torch.ones(3, 4, 7)
    mask[0, 1:] = 0; mask[1, 3] = 0; mask[2] = 0
    mask[1, 1, 2:4] = 0  # holes must not consume prefix probability
    counts = mask.bool().any(-1).sum(-1)
    actor, overlap = renyi_half_and_overlap(q, p)
    # Independent explicit scans, without using the implementation's credit.
    losses = []
    for n in range(3):
        anchors = []
        for a in range(4):
            if not mask[n, a].any(): continue
            reach, prefixes = 1., []
            for d in range(7):
                if mask[n, a, d]:
                    reach = reach * (.3 + .7 * overlap[n, a, d])
                    prefixes.append((d, reach))
            anchors.append(sum(actor[n, a, d] * sum(v for j, v in prefixes if j >= d)
                               for d, _ in prefixes))
        if anchors: losses.append(sum(anchors) / len(anchors))
    full = evaluate(q, p, mask, "sequence_anchor_mean", counts)
    torch.testing.assert_close(full.loss, sum(losses) / len(losses))
    native = evaluate(q, p, mask)
    assert not torch.isclose(full.loss, native.loss)
    chunks = [evaluate(q[:, i:i+1], p[:, i:i+1], mask[:, i:i+1], "sequence_anchor_mean", counts)
              for i in range(4)]
    torch.testing.assert_close(sum(x.loss_terms.numerator for x in chunks) / full.loss_terms.denominator, full.loss)
    assert full.loss_terms.denominator == 2
    for key, terms in full.ratio_metrics.items():
        torch.testing.assert_close(sum(x.ratio_metrics[key].numerator for x in chunks), terms.numerator)
        torch.testing.assert_close(sum(x.ratio_metrics[key].denominator for x in chunks), terms.denominator)
    with pytest.raises(ValueError, match="sample_anchor_counts"):
        evaluate(q, p, mask, "sequence_anchor_mean")


@pytest.mark.parametrize("change", [dict(rho=-.1), dict(rho=1.1), dict(rho=True), dict(rho=float('nan')),
    dict(rho=float('inf')), dict(loss_decay_gamma=4), dict(reduction="weight_mean")])
def test_bad_config(change):
    with pytest.raises(ValueError): normalize_objective(dict(type="dpal_renyi", **change))


def test_config_block_semantics_and_old_defaults():
    assert normalize_objective(dict(type="dpal_renyi")) == dict(
        type="dpal_renyi", rho=.3, reduction="complete_block_mean", loss_decay_gamma=None)
    for kind in ("hard_ce", "soft_ce", "forward_kl", "tv", "l1", "dpace", "bv", "opd_survival"):
        assert normalize_objective(dict(type=kind))["loss_decay_gamma"] == 4.
    with pytest.raises(ValueError, match="token-normalized"):
        normalize_objective(dict(type="weighted_sum", components=[dict(type="dpal_renyi", weight=1)]))


@pytest.mark.parametrize('counts', [torch.tensor([float('nan')]), torch.tensor([-.1]), torch.tensor([.5]), torch.tensor([0.])])
def test_sequence_counts_cannot_hide_invalid_or_missing_anchors(counts):
    with pytest.raises(ValueError, match='sample_anchor_counts'):
        evaluate(torch.randn(1, 1, 7, 5), torch.randn(1, 1, 7, 5), torch.ones(1, 1, 7),
                 'sequence_anchor_mean', counts)


def test_invalid_mask_and_missing_teacher_rejected():
    q, p, mask = torch.randn(1, 1, 7, 5), torch.randn(1, 1, 7, 5), torch.ones(1, 1, 7)
    mask[..., 1] = .5
    with pytest.raises(ValueError, match='binary'): evaluate(q, p, mask)
    with pytest.raises(ValueError, match='teacher'): evaluate(q, None, mask)


def config(proposal="parallel", reduction="complete_block_mean", base_type="hard_ce"):
    obj = dict(type="dpal_renyi", rho=.3, reduction=reduction, loss_decay_gamma=None)
    prop = dict(type=proposal)
    if proposal != "parallel": prop["rank"] = 4
    if "selector" in proposal:
        prop["top_k"] = 3
        obj = dict(type="selector_losses", loss_decay_gamma=None,
                   base=dict(type=base_type, weight=.1), final={k: v for k, v in obj.items() if k != "loss_decay_gamma"})
        obj["final"]["weight"] = 1.
    recipe = dict(context=dict(type="dflash", target_layer_ids=[0, 1, 2]),
        backbone=dict(type="layer_stack", layers=[dict(attention="gqa", access="dense")]),
        proposal=prop, objective=obj, attention_configs=dict(gqa={}), access_configs={})
    return ComposableDraftConfig(composable_config=recipe, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_target_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, vocab_size=19, block_size=8, execution_config=dict(attention_backend="sdpa"), attention_dropout=0.)


def fixture(proposal="parallel", reduction="complete_block_mean", base_type="hard_ce"):
    c = config(proposal, reduction, base_type)
    model = ComposableDraftModel(c)
    embed = nn.Embedding(19, 16).requires_grad_(False)
    head = nn.Linear(16, 19, bias=False).requires_grad_(False)
    ids = torch.randint(19, (3, 14))
    mask = torch.ones_like(ids, dtype=torch.float32); mask[0, 10:] = 0; mask[2] = 0
    data = dict(input_ids=ids, loss_mask=mask, target_hidden_states=torch.randn(3, 14, 48),
                target_last_hidden_states=torch.randn(3, 14, 16))
    layout = prepare_layout(ids, mask, 8, 12, complete_blocks=reduction == "complete_block_mean",
                            supervision_mode="all")
    train = ComposableTrainingModel(model, head, embed, 0, objective_chunk_blocks=1)
    return model, train, data, layout


@pytest.mark.parametrize("proposal", ["parallel", "markov", "rnn", "sequential_rnn", "residual_selector"])
@pytest.mark.parametrize("reduction", ["complete_block_mean", "sequence_anchor_mean"])
def test_real_wrapper_checkpoint_chunks_metrics_and_teacher_alignment(proposal, reduction):
    model, train, data, layout = fixture(proposal, reduction)
    captured = []
    loss = train.loss_branches.objectives['final']
    hook = loss.register_forward_pre_hook(lambda m, args: captured.append((args[0].draft_logits.detach(), args[1])))
    train.objective_chunk_blocks = 0
    full = train(**data, layout=layout)
    logits, supervision = captured[0]
    # B8 supplies D7 already; no second anchor exclusion. Teacher predicts label
    # from its predecessor, independently checked against the input cache.
    assert logits.shape[2] == 7
    indices = (layout.label_indices - 1).clamp_min(0)
    expected_hidden = data['target_last_hidden_states'].gather(1, indices.flatten(1)[..., None].expand(-1, -1, 16))
    torch.testing.assert_close(supervision.target_logits, train.lm_head(expected_hidden).reshape_as(logits))
    hook.remove()
    params = tuple(model.parameters())
    expected_grad = torch.autograd.grad(full.loss, params)
    for chunk in (1, 2, 32):
        train.objective_chunk_blocks = chunk
        part = train(**data, layout=layout)
        torch.testing.assert_close(part.loss, full.loss, rtol=2e-5, atol=2e-6)
        for a, b in zip(torch.autograd.grad(part.loss, params), expected_grad):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, rtol=8e-4, atol=3e-6)
        assert part.ratio_metrics.keys() == full.ratio_metrics.keys()
        for name, ref in full.ratio_metrics.items():
            got = part.ratio_metrics[name]
            torch.testing.assert_close(got.numerator, ref.numerator, rtol=2e-5, atol=1e-5)
            torch.testing.assert_close(got.denominator, ref.denominator)
    active = layout.prediction_mask.bool()
    if reduction == 'complete_block_mean': active = active & active.all(-1, keepdim=True)
    assert full.ratio_metrics['final/dpal_renyi/overlap'].denominator == active.sum()
    assert train.lm_head.weight.grad is None


@pytest.mark.parametrize("reduction", ["complete_block_mean", "sequence_anchor_mean"])
@pytest.mark.parametrize("proposal", ["parallel", "residual_selector"])
def test_uneven_microbatch_window_matches_whole_batch(proposal, reduction):
    model, train, data, layout = fixture(proposal, reduction)
    whole = train(**data, layout=layout)
    expected = torch.autograd.grad(whole.loss, tuple(model.parameters()))
    den = train.loss_branches.native_denominators(layout)
    window = (torch.stack([torch.tensor(3.), den['final'], den['base']]) if train.selector_enabled else None)
    parts = [train(**{k: v[i:i+1] for k, v in data.items()}, layout=layout.batch_slice(i, i+1),
                   window_denominators=window) for i in range(3)]
    combined = sum(x.loss_terms.numerator for x in parts) / sum(x.loss_terms.denominator for x in parts)
    torch.testing.assert_close(combined, whole.loss, rtol=1e-5, atol=2e-6)
    for a, b in zip(torch.autograd.grad(combined, tuple(model.parameters())), expected):
        torch.testing.assert_close(a, b, rtol=8e-4, atol=3e-6)


def test_selector_actual_final_distribution_candidate_policy_and_inactive_teacher():
    branches = LossBranches(config('residual_selector').composable_config)
    base, final, p = (torch.randn(1, 2, 7, 19, requires_grad=True) for _ in range(3))
    mask = torch.ones(1, 2, 7)
    sup = Supervision(torch.zeros_like(mask, dtype=torch.long), mask, p)
    pred = DraftOutput(base, draft_logits=final, base_logits=base, candidate_ids=torch.zeros(1, 2, 7, 3, dtype=torch.long))
    got = branches.branch_output('final', pred, sup)
    ref = evaluate(final, p, mask)
    torch.testing.assert_close(got.loss, ref.loss)
    assert not torch.isclose(got.loss, evaluate(base, p, mask).loss)
    assert torch.autograd.grad(got.loss, base, allow_unused=True)[0] is None
    with pytest.raises(ValueError, match='candidate CE'):
        config('topk_selector')
    recipe = deepcopy(branches.recipe)
    recipe['objective']['final'] = dict(type='hard_ce', weight=1.)
    recipe['objective']['base'] = dict(type='dpal_renyi', weight=0.)
    assert not LossBranches(recipe).needs_teacher
    recipe['proposal']['type'] = 'topk_selector'; recipe['proposal'].pop('selector_mode')
    recipe['objective']['base']['weight'] = .1
    topk = LossBranches(recipe)
    pred.candidate_logits = torch.randn(1, 2, 7, 3)
    # Full-vocabulary base remains legal even when final support is restricted.
    torch.testing.assert_close(topk.branch_output('base', pred, sup).loss, evaluate(base, p, mask).loss)


def test_export_and_config_roundtrip_without_training_dependency():
    from transformers import AutoModel
    model, _, _, _ = fixture('residual_selector', 'sequence_anchor_mean')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
        export_composable(model, folder)
        loaded = AutoModel.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
        assert loaded.config.composable_config == model.config.composable_config
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, loaded.state_dict()[k])
        assert not any('core/composable/objectives' in str(p) for p in Path(folder).rglob('*.py'))


def test_new_examples_and_old_presets_validate():
    for file in sorted((ROOT / 'configs').glob('*.json')):
        value = json.loads(file.read_text())
        if value.get('schema_version') != 'composable_v3': continue
        value.setdefault('num_target_layers', 36)
        if file.name == '25swa_dpal_qwen35_9b.json':
            # Recipe-only JSON inherits these from its Qwen3.5-9B target in
            # the launcher; Qwen3Config's default 32 Q heads is not that target.
            value.update(num_attention_heads=16, num_key_value_heads=4, head_dim=256)
        ComposableDraftConfig(**value)


def test_training_metadata_records_science_and_source_identity():
    from litespec.training.recipes.composable import ComposableTrainingAdapter
    model, _, _, _ = fixture()
    args = SimpleNamespace(objective_chunk_blocks=32, num_anchors=512, seed=42,
                           complete_blocks=False, bv_sampling_provenance=None)
    metadata = ComposableTrainingAdapter('composable', 'composable').training_metadata(
        SimpleNamespace(draft=model, args=args))
    assert metadata['loss_config']['type'] == 'dpal_renyi'
    assert metadata['loss_contract_version'] == 2
    assert metadata['dpal_reference_commit'] == '23a628bd9d65b56c21f99692003ac5a3f844dd17'
    assert metadata['dpal_source_sha256'] == hashlib.sha256(
        (ROOT / 'litespec/core/composable/objectives/dpal_renyi.py').read_bytes()).hexdigest()
    assert metadata['dpal_interpretation'] == 'offline_teacher_forced_surrogate'


@pytest.mark.parametrize('kind', ['hard_ce', 'soft_ce', 'forward_kl', 'tv', 'l1', 'dpace', 'bv', 'tv_survival', 'opd_survival'])
def test_old_wrapper_loss_gradients_and_metrics_unchanged(kind):
    file = ROOT / 'tests/reference/dpal_before/litespec/core/composable/training_model.py'
    spec = importlib.util.spec_from_file_location('litespec.core.composable._before_dpal', file)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    model, train, data, layout = fixture()
    model.config.composable_config['objective'] = dict(type=kind, loss_decay_gamma=4.)
    train.loss_branches = LossBranches(model.config.composable_config)
    train.component_normalization = kind == 'opd_survival'
    previous = old.ComposableTrainingModel(model, train.lm_head, train.embed_tokens, 0, objective_chunk_blocks=1)
    a, b = train(**data, layout=layout), previous(**data, layout=layout)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    for x, y in zip(torch.autograd.grad(a.loss, tuple(model.parameters())),
                    torch.autograd.grad(b.loss, tuple(model.parameters()))):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert a.ratio_metrics.keys() == b.ratio_metrics.keys()
    for key in a.ratio_metrics:
        torch.testing.assert_close(a.ratio_metrics[key].numerator, b.ratio_metrics[key].numerator, rtol=0, atol=0)
        torch.testing.assert_close(a.ratio_metrics[key].denominator, b.ratio_metrics[key].denominator, rtol=0, atol=0)


@pytest.mark.parametrize('reduction', ['complete_block_mean', 'sequence_anchor_mean'])
def test_dpal_with_opd_sibling_keeps_component_and_metric_denominators(reduction):
    model, train, data, layout = fixture('residual_selector', reduction, base_type='opd_survival')
    train.objective_chunk_blocks = 0
    full = train(**data, layout=layout)
    expected = torch.autograd.grad(full.loss, tuple(model.parameters()))
    train.objective_chunk_blocks = 1
    part = train(**data, layout=layout)
    torch.testing.assert_close(part.loss, full.loss, rtol=1e-5, atol=2e-6)
    for x, y in zip(torch.autograd.grad(part.loss, tuple(model.parameters())), expected):
        torch.testing.assert_close(x, y, rtol=8e-4, atol=3e-6)
    for key in full.ratio_metrics:
        torch.testing.assert_close(part.ratio_metrics[key].numerator, full.ratio_metrics[key].numerator, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(part.ratio_metrics[key].denominator, full.ratio_metrics[key].denominator)
    metadata = train(**data, layout=layout, metadata_only=True)
    native = train.loss_branches.native_denominators(layout)['final']
    torch.testing.assert_close(metadata['component_denominators']['final']['token'], native)

