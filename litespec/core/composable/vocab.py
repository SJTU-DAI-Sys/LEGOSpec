"""Forward-local vocabulary intermediates shared by compatible loss branches."""

import torch
import torch.nn.functional as F


class VocabDistribution:
    def __init__(self, logits):
        self.logits = logits
        self._float = None
        self._probabilities = None
        self._log_probabilities = None
        self._exp_log_probabilities = None
        self._cross_entropies = {}

    def float_logits(self):
        if self._float is None:
            self._float = self.logits.float()
        return self._float

    def probabilities(self):
        if self._probabilities is None:
            # Keep native softmax rounding. exp(log_softmax) changes the L1
            # subgradient when low-precision draft and teacher rows coincide.
            self._probabilities = torch.softmax(self.float_logits(), dim=-1)
            self._float = None
        return self._probabilities

    def log_probabilities(self):
        if self._log_probabilities is None:
            self._log_probabilities = F.log_softmax(self.float_logits(), dim=-1)
            self._float = None
        return self._log_probabilities

    def exp_log_probabilities(self):
        if self._exp_log_probabilities is None:
            self._exp_log_probabilities = self.log_probabilities().exp()
        return self._exp_log_probabilities

    def cross_entropy(self, labels):
        key = id(labels)
        if key not in self._cross_entropies:
            loss = F.cross_entropy(self.float_logits().flatten(0, -2), labels.flatten(),
                                   reduction="none").reshape_as(labels)
            self._cross_entropies[key] = (labels, loss)
            self._float = None
        return self._cross_entropies[key][1]


class VocabCache:
    """Owned by one loss invocation, never retained across checkpoint replay."""

    def __init__(self, frozen_entries=None):
        self._entries = {}
        self._frozen_entries = {} if frozen_entries is None else frozen_entries

    def distribution(self, logits, *, frozen=False):
        entries = self._frozen_entries if frozen else self._entries
        key = id(logits)
        if key not in entries:
            entries[key] = VocabDistribution(logits.detach() if frozen else logits)
        return entries[key]

    def fork_draft(self):
        # Each low-precision objective must retain its own cast-back gradient
        # boundary. Frozen teacher intermediates can still be shared exactly.
        return VocabCache(self._frozen_entries)

    def detached_probabilities(self, logits):
        existing = self._entries.get(id(logits))
        if existing is not None and existing._probabilities is not None:
            return existing._probabilities.detach()
        key = id(logits)
        if key not in self._frozen_entries and existing is not None and existing._float is not None:
            self._frozen_entries[key] = VocabDistribution(existing._float.detach())
        return self.distribution(logits, frozen=True).probabilities()

    def peek_detached_probabilities(self, logits):
        """Reuse existing rows for metrics without allocating full-vocab buffers."""
        for entries in (self._entries, self._frozen_entries):
            value = entries.get(id(logits))
            if value is not None and value._probabilities is not None:
                return value._probabilities.detach()
        return None

    def register_indexed_teacher(self, expanded, unique, rows):
        value = IndexedTeacherDistribution(unique, rows)
        self._frozen_entries[id(expanded)] = value
        return value


class IndexedTeacherDistribution(VocabDistribution):
    """Normalize frozen source rows once, then expand only for each consumer."""

    def __init__(self, logits, rows):
        super().__init__(logits)
        self.rows = rows
        self.unique_distribution = VocabDistribution(logits)

    def _expand(self, value):
        return value.index_select(0, self.rows.flatten()).reshape(*self.rows.shape, -1)

    def probabilities(self):
        if self._probabilities is None:
            self._probabilities = self._expand(self.unique_distribution.probabilities())
        return self._probabilities

    def log_probabilities(self):
        if self._log_probabilities is None:
            self._log_probabilities = self._expand(self.unique_distribution.log_probabilities())
        return self._log_probabilities

    def exp_log_probabilities(self):
        if self._exp_log_probabilities is None:
            self._exp_log_probabilities = self._expand(self.unique_distribution.exp_log_probabilities())
        return self._exp_log_probabilities


def distribution(logits, supervision, *, frozen=False):
    cache = getattr(supervision, "vocab_cache", None)
    if cache is None:
        return VocabDistribution(logits.detach() if frozen else logits)
    return cache.distribution(logits, frozen=frozen)


@torch.no_grad()
def project_teacher_rows(head, hidden, indices, *, active=None, cache=None, row_plan=None):
    """Project each requested (sample, position) once, preserving row order."""
    batch, length, width = hidden.shape
    positions = indices + torch.arange(batch, device=indices.device)[:, None, None] * length
    if row_plan is None:
        flat_positions = positions.flatten() if active is None else positions[active]
        unique, inverse = torch.unique(flat_positions, sorted=True, return_inverse=True)
    else:
        unique, inverse = row_plan.unique_teacher_index, row_plan.teacher_inverse
    selected = hidden.reshape(batch * length, width).index_select(0, unique)
    logits = head(selected)
    if active is None:
        return logits.index_select(0, inverse).reshape(*indices.shape, logits.size(-1))
    # Row zero is a harmless uniform distribution for unsupervised slots.
    logits = F.pad(logits, (0, 0, 1, 0))
    rows = (torch.zeros_like(indices).masked_scatter(active, inverse + 1) if row_plan is None else
            torch.zeros_like(indices).flatten().index_copy(0, row_plan.active_index, inverse + 1).reshape_as(indices))
    expanded = logits.index_select(0, rows.flatten()).reshape(*indices.shape, logits.size(-1))
    if cache is not None:
        cache.register_indexed_teacher(expanded, logits, rows)
    return expanded
