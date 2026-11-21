"""Full-vocabulary fused reductions preserve native softmax gradients."""
import pytest
import torch
import torch.nn.functional as F
from litespec.core.composable.objectives.soft_ce import _probability_log_dot
from litespec.core.losses import _run

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA fusion validation')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('kl',[False,True])
def test_native_softmax_gradient_boundaries(dtype,kl,monkeypatch):
    monkeypatch.setenv('LITESPEC_LOSS_BACKEND','inductor')
    torch.manual_seed(42)
    q=torch.randn(71,8192,device='cuda',dtype=dtype,requires_grad=True)
    ref=q.detach().clone().requires_grad_()
    p=torch.randn_like(q).float().log_softmax(-1)
    probs=p.exp();weights=torch.rand(71,device='cuda');weights[::3]=0
    log_q=F.log_softmax(q.float(),-1)
    actual=-_run(_probability_log_dot,probs,log_q)
    expected=-(probs*F.log_softmax(ref.float(),-1)).sum(-1)
    if kl:
        actual=actual+_run(_probability_log_dot,probs,p)
        expected=expected+(probs*p).sum(-1)
    actual=(actual*weights).sum();expected=(expected*weights).sum()
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-5)
    actual.backward();expected.backward()
    torch.testing.assert_close(q.grad,ref.grad,rtol=0,atol=0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation validation')
def test_fusion_avoids_full_vocab_product(monkeypatch):
    monkeypatch.setenv('LITESPEC_LOSS_BACKEND','inductor')
    p=torch.rand(512,32768,device='cuda');q=torch.rand_like(p)
    _run(_probability_log_dot,p,q);torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    actual=_run(_probability_log_dot,p,q);torch.cuda.synchronize()
    fused_peak=torch.cuda.max_memory_allocated()-base
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    expected=(p*q).sum(-1);torch.cuda.synchronize()
    eager_peak=torch.cuda.max_memory_allocated()-base
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-5)
    assert fused_peak<eager_peak//4,(fused_peak,eager_peak)
