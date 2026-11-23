"""Select draft context and normalized teacher features from target outputs."""

import torch


def hidden_states_to_cuda(hidden_states):
    if isinstance(hidden_states, dict):
        return {
            key: value.cuda(non_blocking=True) for key, value in hidden_states.items()
        }
    return hidden_states.cuda(non_blocking=True)


def concatenate_target_hidden(
    hidden_states: dict[int, torch.Tensor], target_layer_ids: list[int]
):
    return torch.cat([hidden_states[layer] for layer in target_layer_ids], dim=-1)


def select_target_features(
    hidden_states, final_hidden_state, target_layer_ids, require_last
):
    context = concatenate_target_hidden(hidden_states, target_layer_ids)
    if require_last and final_hidden_state is None:
        raise ValueError(
            "Teacher supervision requires explicit post-final-norm hidden states"
        )
    return context, final_hidden_state if require_last else None


def prepare_target_features(data, target_output, draft, require_last: bool):
    hidden_states = (
        data["hidden_states"] if target_output is None else target_output.hidden_states
    )
    final_hidden = (
        data.get("final_hidden_state")
        if target_output is None
        else target_output.final_hidden_state
    )
    hidden_states = hidden_states_to_cuda(
        {layer: hidden_states[layer] for layer in draft.target_layer_ids}
    )
    context, teacher = select_target_features(
        hidden_states,
        final_hidden,
        draft.target_layer_ids,
        require_last,
    )
    return context, hidden_states_to_cuda(teacher) if teacher is not None else None
