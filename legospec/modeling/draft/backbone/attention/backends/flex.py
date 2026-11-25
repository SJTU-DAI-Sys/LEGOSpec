"""Shared FlexAttention runtime for every draft algorithm."""

from __future__ import annotations

import torch
import torch._dynamo as dynamo
from packaging.version import Version
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.utils import is_torchdynamo_compiling

from legospec.modeling.draft.backbone.attention.backends.torch_compat import (
    patch_inductor_cutedsl_lowerings,
)

dynamo.config.recompile_limit = 64

FLASH_Q_TILE = 256
FLASH_KV_TILE = 128
TRITON_MASK_TILE = 128


def flex_kernel_options(backend: str) -> dict:
    """Translate the public backend name into version-safe kernel options."""

    backend = str(backend).upper()
    version = Version(torch.__version__.split("+", 1)[0])
    if backend == "FLASH":
        if version < Version("2.11"):
            raise RuntimeError("FLASH FlexAttention training requires torch >= 2.11")
        patch_inductor_cutedsl_lowerings()
        return {"BACKEND": "FLASH"}
    if backend == "TRITON":
        return (
            {"BACKEND": "TRITON"}
            if version >= Version("2.11")
            else {"FORCE_USE_FLEX_ATTENTION": True}
        )
    raise ValueError("FlexAttention backend must be TRITON or FLASH")


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled:
            # Enable dynamic shapes to handle different input sizes
            self._compiled_flex_attention = torch.compile(
                flex_attention,
                # mode="max-autotune-no-cudagraphs",
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


class WrappedStaticFlexAttention:
    """Static-shape compiler required by the CuteDSL FLASH backend.

    ``torch.compile`` still caches one specialization per physical shape.  The
    wrapper only disables symbolic dimensions, which FLASH cannot inline when
    a block mask captures shape-dependent scalar values.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_flex_compiled:
            self._compiled_flex_attention = torch.compile(
                flex_attention,
                dynamic=False,
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    kernel_options = kwargs.get("kernel_options") or {}
    needs_static_shapes = kernel_options.get("BACKEND") == "FLASH"
    if is_torchdynamo_compiling():
        flex_attention_compiled = flex_attention
    elif needs_static_shapes:
        flex_attention_compiled = WrappedStaticFlexAttention()()
    else:
        flex_attention_compiled = WrappedFlexAttention()()
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


def create_flex_block_mask(
    mask_mod,
    *,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
    backend: str,
):
    """Create a BlockMask with one backend policy for every algorithm."""

    backend = str(backend).upper()
    # Validate the backend and apply the FLASH compatibility patch through the
    # same path used to construct attention kernel options.
    flex_kernel_options(backend)
    # Torch 2.13 Inductor cannot lower data-dependent sparse block counts for
    # packed masks. Build BlockMask eagerly; attention kernels stay compiled.
    block_size = (
        (FLASH_Q_TILE, FLASH_KV_TILE) if backend == "FLASH" else TRITON_MASK_TILE
    )
    return create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
        BLOCK_SIZE=block_size,
    )


__all__ = [
    "FLASH_KV_TILE",
    "FLASH_Q_TILE",
    "compile_friendly_flex_attention",
    "create_flex_block_mask",
    "flex_kernel_options",
]
