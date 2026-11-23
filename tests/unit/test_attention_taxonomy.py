import json
from pathlib import Path
import pytest
import torch
from legospec.modeling.draft import LegoSpecConfig, LegoSpecModel
from legospec.modeling.draft.backbone.state import DraftState
from legospec.modeling.draft.contracts import DraftInputs
from legospec.modeling.draft.export import export_composable


def config(variant="gqa", access="full"):
    family = (
        "linear"
        if variant in ("gdn", "kda")
        else "latent"
        if variant == "mla"
        else "dense"
    )
    a = dict(family=family, variant=variant)
    if family != "linear":
        a["access"] = access
    profile = {}
    if family == "linear":
        profile = dict(
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            query_conv_kernel_size=2,
            context_conv_kernel_size=2,
        )
    if variant == "mla":
        profile = dict(
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
        )
    ac = (
        {"dsa": dict(num_query_heads=2, head_dim=8, rope_head_dim=4, index_topk=2)}
        if access == "dsa"
        else {}
    )
    return LegoSpecConfig(
        hidden_size=16,
        intermediate_size=32,
        vocab_size=31,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_target_layers=4,
        block_size=4,
        execution_config={"gdn_backend": "reference"},
        composable_config=dict(
            context=dict(type="concat_projection", target_layer_ids=[0, 2]),
            backbone=dict(type="layer_stack", layers=[dict(attention=a)]),
            attention_configs={variant: profile},
            access_configs=ac,
            proposal=dict(type="parallel"),
            objective=dict(type="hard_ce"),
        ),
    )


def data():
    return DraftInputs(
        target_hidden_states=torch.randn(1, 7, 32),
        draft_embeddings=torch.randn(1, 8, 16),
        anchor_positions=torch.tensor([[3, 6]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        mode="features",
    )


@pytest.mark.parametrize("variant", ["mha", "gqa", "mqa", "mla", "gdn", "kda"])
def test_forward_backward_roundtrip(variant, tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    c = config(variant)
    m = LegoSpecModel(c)
    x = data()
    y = m(x).backbone_hidden_states
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    if variant in ("gdn", "kda"):
        block = m.backbone.layers[0].attention.block_attention
        assert block.gate_proj.weight.grad is not None
        assert m.backbone.layers[0].attention.mixer.in_proj_a.weight.grad is not None
        assert "block_attention" not in json.dumps(c.to_dict())
    c.save_pretrained(tmp_path)
    restored = LegoSpecConfig.from_pretrained(tmp_path)
    assert restored.composable_config == c.composable_config
    m.save_pretrained(tmp_path)
    loaded, info = LegoSpecModel.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    torch.testing.assert_close(loaded(x).backbone_hidden_states, y)


@pytest.mark.parametrize("variant", ["mha", "gqa", "mqa", "mla"])
@pytest.mark.parametrize("access", ["full", "swa", "dsa"])
def test_token_access(variant, access, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    c = config(variant, access)
    m = LegoSpecModel(c).eval()
    x = data()
    y = m(x).backbone_hidden_states
    assert torch.isfinite(y).all()
    if access == "dsa":
        from dsa_reference import reference_attention
        from legospec.modeling.draft.backbone.attention.access import (
            dsa as sparse,
        )

        monkeypatch.setattr(sparse, "gather_attention", reference_attention)
        ref = LegoSpecModel(c).eval()
        ref.load_state_dict(m.state_dict())
        torch.testing.assert_close(
            ref(x).backbone_hidden_states, y, atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_linear_cache_matches_full_prefix(variant):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    m = LegoSpecModel(config(variant)).eval()
    context = torch.randn(1, 7, 32)
    emb = torch.randn(1, 4, 16)

    def run(c, positions, anchor, state=None):
        x = DraftInputs(
            c,
            emb,
            torch.tensor([[anchor]]),
            torch.ones(1, 1, dtype=torch.bool),
            context_position_ids=positions,
            draft_state=state,
            mode="features",
        )
        return m(x)

    state = DraftState()
    run(context[:, :3], torch.arange(3)[None], 3, state)
    cached = run(context[:, 3:6], torch.arange(3, 6)[None], 6, state)
    full = run(context[:, :6], torch.arange(6)[None], 6)
    torch.testing.assert_close(
        cached.backbone_hidden_states, full.backbone_hidden_states, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("field,value", [("block_attention", {}), ("access", "full")])
def test_reject_legacy_layer_options(field, value):
    c = config("gdn")
    c.composable_config["backbone"]["layers"][0][field] = value
    with pytest.raises(ValueError):
        c.validate_recipe()


def test_reject_linear_token_access():
    c = config("kda")
    c.composable_config["backbone"]["layers"][0]["attention"]["access"] = "full"
    with pytest.raises(ValueError):
        c.validate_recipe()


def test_all_active_configs():
    root = Path(__file__).resolve().parents[2]
    for p in (root / "configs").glob("*.json"):
        d = json.loads(p.read_text())
        if "composable_config" not in d:
            continue
        profiles = d["composable_config"]["attention_configs"]
        dense = next((profiles[k] for k in ("gqa", "mha", "mqa") if k in profiles), {})
        c = LegoSpecConfig(
            hidden_size=2560,
            num_attention_heads=dense.get("num_attention_heads", 32),
            num_key_value_heads=dense.get("num_key_value_heads", 8),
            head_dim=dense.get("head_dim", 128),
            num_target_layers=36,
            **{k: v for k, v in d.items() if k != "architectures"},
        )
        c.runtime_config()


@pytest.mark.parametrize("variant", ["gdn", "kda", "mla"])
def test_standalone_export(variant, tmp_path):
    from transformers import AutoModel

    torch.set_num_threads(1)
    m = LegoSpecModel(config(variant)).eval()
    x = data()
    export_composable(m, tmp_path)
    loaded = AutoModel.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    ).eval()
    # Dynamic modules have separate dataclass identities; a fresh input is enough
    # for noncached execution, which uses the same documented field contract.
    torch.testing.assert_close(
        loaded(x).backbone_hidden_states, m(x).backbone_hidden_states
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_linear_fla_reference_outputs_state_and_gradients(variant):
    if variant == "kda":
        from fla.ops.kda import chunk_kda as scan
    else:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as scan
    from legospec.modeling.draft.backbone.attention.linear.recurrence import (
        reference_event_scan,
    )

    torch.manual_seed(42)
    torch.set_num_threads(1)
    q = torch.randn(
        1, 24, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(
        1, 24, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    shape = (1, 24, 2, 32) if variant == "kda" else (1, 24, 2)
    g = (-torch.rand(*shape, device="cuda") * 0.1).requires_grad_()
    b = torch.rand(1, 24, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Identity read events: no decay or writes, including consecutive block slots.
    write = torch.ones(1, 24, 1, device="cuda")
    write[:, 8:12] = 0
    write[:, 20:24] = 0
    masked_g = g * (write[..., None] if variant == "kda" else write)
    masked_b = b * write
    boundaries = torch.tensor([0, 24], dtype=torch.int32)
    initial = torch.randn(1, 2, 32, 32, device="cuda") * 0.01
    expected, state = reference_event_scan(
        q, k, v, masked_g, masked_b, boundaries, initial
    )
    actual, last = scan(
        q,
        k,
        v,
        masked_g,
        masked_b,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
    torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.05)
    torch.testing.assert_close(last, state, atol=0.015, rtol=0.05)
    tensors = (q, k, v, g, b)
    grad1 = torch.autograd.grad(
        actual.float().square().sum(), tensors, retain_graph=True
    )
    grad2 = torch.autograd.grad(expected.float().square().sum(), tensors)
    for a, e in zip(grad1, grad2):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, atol=0.015, rtol=0.1)


def test_mla_dsa_indexer_gradient_and_cached_prefix():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    m = LegoSpecModel(config("mla", "dsa"))
    x = data()
    x.collect_indexer_teacher = True
    out = m(x)
    aux = out.indexer_aux
    assert aux is not None and aux.denominator > 0
    (
        out.backbone_hidden_states.square().mean() + aux.numerator / aux.denominator
    ).backward()
    for p in m.backbone.indexer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    m.eval()
    state = DraftState()
    context = torch.randn(1, 6, 32)
    emb = torch.randn(1, 4, 16)

    def run(start, end, state):
        return m(
            DraftInputs(
                context[:, start:end],
                emb,
                torch.tensor([[end]]),
                torch.ones(1, 1, dtype=torch.bool),
                context_position_ids=torch.arange(start, end)[None],
                draft_state=state,
                mode="features",
            )
        )

    run(0, 3, state)
    cached = run(3, 6, state)
    full = run(0, 6, None)
    torch.testing.assert_close(
        cached.backbone_hidden_states, full.backbone_hidden_states, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("variant", ["gdn", "kda"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_model_fla_forward_backward_and_cache(variant):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    c = config(variant)
    p = c.composable_config["attention_configs"][variant]
    p.update(linear_key_head_dim=32, linear_value_head_dim=32)
    c.execution_config["gdn_backend"] = "fla"
    model = LegoSpecModel(c).cuda().to(torch.bfloat16)
    x = data()
    x.target_hidden_states = x.target_hidden_states.cuda().bfloat16()
    x.draft_embeddings = x.draft_embeddings.cuda().bfloat16()
    x.anchor_positions = x.anchor_positions.cuda()
    x.block_keep_mask = x.block_keep_mask.cuda()
    y = model(x).backbone_hidden_states
    y.float().square().mean().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(model.backbone.layers[0].attention.mixer.in_proj_a.weight.grad).all()
    assert torch.isfinite(
        model.backbone.layers[0].attention.block_attention.gate_proj.weight.grad
    ).all()
    model.eval()
    state = DraftState()
    context = x.target_hidden_states[:, :6]
    emb = x.draft_embeddings[:, :4]

    def run(start, end, state):
        return model(
            DraftInputs(
                context[:, start:end],
                emb,
                torch.tensor([[end]], device="cuda"),
                torch.ones(1, 1, device="cuda", dtype=torch.bool),
                context_position_ids=torch.arange(start, end, device="cuda")[None],
                draft_state=state,
                mode="features",
            )
        ).backbone_hidden_states

    with torch.no_grad():
        run(0, 3, state)
        cached = run(3, 6, state)
        full = run(0, 6, None)
    torch.testing.assert_close(cached, full, atol=0.015, rtol=0.05)


def test_mla_dsa_absorbed_cache_matches_expanded():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = LegoSpecModel(config("mla", "dsa")).eval()
    x = data()
    expanded = model(x).backbone_hidden_states
    with torch.no_grad():
        absorbed = model(x).backbone_hidden_states
    torch.testing.assert_close(absorbed, expanded, atol=2e-6, rtol=2e-5)
    state = DraftState()
    context = torch.randn(1, 6, 32)
    emb = torch.randn(1, 4, 16)

    def run(start, end, state):
        return model(
            DraftInputs(
                context[:, start:end],
                emb,
                torch.tensor([[end]]),
                torch.ones(1, 1, dtype=torch.bool),
                context_position_ids=torch.arange(start, end)[None],
                draft_state=state,
                mode="features",
            )
        ).backbone_hidden_states

    with torch.no_grad():
        run(0, 3, state)
        cached = run(3, 6, state)
        full = run(0, 6, None)
    torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-5)


def test_mla_optimizer_head_geometry_after_module_move():
    from legospec.training.optim.optimizer import register_optimizer_parameters

    model = LegoSpecModel(config("mla"))
    parameters, specs = register_optimizer_parameters(model)
    assert len(parameters) == len(list(model.parameters()))
    by_name = {spec.name: spec for spec in specs}
    query = by_name["backbone.layers.0.attention.q_b_proj.weight"]
    key_value = by_name["backbone.layers.0.attention.kv_b_proj.weight"]
    assert (query.optimizer, query.muon_heads, query.muon_head_dims) == (
        "muon",
        2,
        (8,),
    )
    assert (key_value.optimizer, key_value.muon_heads, key_value.muon_head_dims) == (
        "muon",
        2,
        (4, 8),
    )
