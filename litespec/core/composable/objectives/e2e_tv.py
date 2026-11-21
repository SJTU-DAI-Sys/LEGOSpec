"""End-to-end expected prefix survival over complete draft blocks."""

import torch

from .base import Objective


def token_overlap(draft_logits, target_logits, complete):
    # Mask invalid blocks before the softmax so padding cannot poison gradients.
    keep = complete[..., None, None]
    q = torch.where(keep, draft_logits, 0).float().softmax(-1)
    p = torch.where(keep, target_logits.detach(), 0).float().softmax(-1)
    return (1 - 0.5 * (p - q).abs().sum(-1)).clamp(0, 1)


# CUDA fuses vocabulary-sized intermediates. The entire anchor batch stays intact.
_compiled_overlap = None


class E2ETVObjective(Objective):
    needs_teacher = True
    requires_complete_blocks = True
    metric_names = ("e2e_tv/expected_prefix",)

    def native_denominator(self, layout):
        return layout.prediction_mask.bool().all(-1).sum().float()

    def forward(self, prediction, supervision):
        draft = prediction.draft_logits
        teacher = supervision.target_logits
        mask = supervision.prediction_mask
        if (draft is None or teacher is None or draft.shape != teacher.shape
                or draft.shape[:-1] != mask.shape or mask.ndim != 3 or not mask.size(-1)
                or prediction.candidate_logits is not None):
            raise ValueError("E2E-TV requires aligned full-vocabulary logits and [N,A,D] supervision")
        complete = mask.bool().all(-1)
        overlap = token_overlap
        if draft.is_cuda:
            global _compiled_overlap
            if _compiled_overlap is None:
                _compiled_overlap = torch.compile(token_overlap, fullgraph=True)
            overlap = _compiled_overlap
        prefix = overlap(draft, teacher, complete).cumprod(-1)
        numerator = ((1 - prefix.mean(-1)) * complete).sum()
        count = complete.sum().to(numerator)
        expected = (prefix.detach().sum(-1) * complete).sum()
        return self.output(numerator, count, **{"e2e_tv/expected_prefix": expected})
