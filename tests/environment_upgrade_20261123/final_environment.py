from pathlib import Path
import importlib.metadata as m,json,sys,torch
from sglang.srt.distributed.bootstrap import init_layer_runtime
import causal_conv1d_cuda
names=["sglang","sglang-kernel","torch","transformers","triton","flash-linear-attention","causal-conv1d","tilelang","flashinfer-python"]
record=dict(executable=sys.executable,prefix=sys.prefix,torch_runtime=torch.__version__,cuda=torch.version.cuda,
 bootstrap=init_layer_runtime.__module__,causal_conv1d=causal_conv1d_cuda.__file__,
 packages={n:m.version(n) for n in names})
(Path(__file__).parent/"environment_after.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
