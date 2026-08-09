from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from .runtime import SpectrumIdeogram4Runtime
from .sampling import (
    ACTUAL_KEY,
    COORDINATE_KEY,
    REASON_KEY,
    RUNTIME_KEY,
    RUN_ID_KEY,
    STEP_ID_KEY,
)

from comfy.ldm.ideogram4.model import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    _split_half_rope_matrix,
)
from comfy.text_encoders.llama import precompute_freqs_cis

LOG = logging.getLogger(__name__)


def locate_native_ideogram4(model: Any) -> tuple[Any | None, str | None]:
    candidates = (
        (getattr(getattr(model, "model", None), "diffusion_model", None), "model.diffusion_model"),
        (getattr(model, "diffusion_model", None), "diffusion_model"),
    )
    for inner, path in candidates:
        if is_native_ideogram4(inner):
            return inner, path
    return None, None


def is_native_ideogram4(inner: Any) -> bool:
    if inner is None:
        return False
    if inner.__class__.__name__ != "Ideogram4Transformer2DModel":
        return False
    if not inner.__class__.__module__.startswith("comfy.ldm.ideogram4.model"):
        return False
    return all(
        hasattr(inner, name)
        for name in (
            "input_proj",
            "llm_cond_norm",
            "llm_cond_proj",
            "t_embedding",
            "adaln_proj",
            "embed_image_indicator",
            "layers",
            "final_layer",
            "_img_to_tokens",
            "_tokens_to_img",
            "_image_position_ids",
        )
    )


def require_native_ideogram4(model: Any) -> Any:
    inner, _ = locate_native_ideogram4(model)
    if inner is None:
        raise RuntimeError(
            "Spectrum Apply Ideogram 4 requires ComfyUI's native "
            "comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel."
        )
    return inner


def _hashable(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _branch_labels(transformer_options: dict[str, Any], batch_size: int) -> tuple[Any, ...] | None:
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    uuids = transformer_options.get("uuids")
    cond = None if cond_or_uncond is None else tuple(_hashable(value) for value in cond_or_uncond)
    ids = None if uuids is None else tuple(_hashable(value) for value in uuids)
    if cond is not None and len(cond) != batch_size:
        cond = None
    if ids is not None and len(ids) != batch_size:
        ids = None
    if cond is not None and ids is not None:
        return tuple((cond[index], ids[index]) for index in range(batch_size))
    if cond is not None:
        return cond
    if ids is not None:
        return tuple(("uuid", value) for value in ids)
    if batch_size == 1:
        return (0,)
    return None


def _model_branch_labels(
    inner: Any,
    transformer_options: dict[str, Any],
    batch_size: int,
) -> tuple[Any, ...] | None:
    labels = _branch_labels(transformer_options, batch_size)
    if labels is None:
        return None
    model_identity = id(inner)
    return tuple(("ideogram4_model", model_identity, label) for label in labels)


def _feature_width(inner: Any) -> int:
    normalized_shape = getattr(getattr(inner.final_layer, "norm_final", None), "normalized_shape", None)
    if normalized_shape is not None:
        return int(normalized_shape[0])
    return int(inner.head_dim * inner.num_heads)


def _topology_signature(
    inner: Any,
    x: torch.Tensor,
    context: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    image_token_count: int,
) -> tuple[Any, ...]:
    context_shape = None if context is None else tuple(int(value) for value in context.shape[1:])
    mask_shape = None if attention_mask is None else tuple(int(value) for value in attention_mask.shape[1:])
    return (
        "ideogram4",
        tuple(int(value) for value in x.shape[1:]),
        context_shape,
        mask_shape,
        int(image_token_count),
        _feature_width(inner),
        int(len(inner.layers)),
        int(inner.head_dim),
        tuple(int(value) for value in inner.mrope_section),
    )


def _packed_inputs(
    inner: Any,
    x: torch.Tensor,
    context: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    gh: int,
    gw: int,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
    batch_size = x.shape[0]
    img_tokens = inner._img_to_tokens(x)
    image_token_count = img_tokens.shape[1]
    device = x.device

    if context is None:
        position_ids = inner._image_position_ids(gh, gw, device).unsqueeze(0).expand(batch_size, image_token_count, 3)
        indicator = torch.full(
            (batch_size, image_token_count), OUTPUT_IMAGE_INDICATOR, dtype=torch.long, device=device
        )
        return None, img_tokens, position_ids, None, indicator, 0

    text_length = context.shape[1]
    sequence_length = text_length + image_token_count
    latent_dim = img_tokens.shape[-1]
    packed = torch.zeros(batch_size, sequence_length, latent_dim, dtype=img_tokens.dtype, device=device)
    packed[:, text_length:] = img_tokens

    text_pos = torch.arange(text_length, device=device).view(-1, 1).expand(text_length, 3)
    image_pos = inner._image_position_ids(gh, gw, device)
    position_ids = torch.cat([text_pos, image_pos], dim=0).unsqueeze(0).expand(batch_size, sequence_length, 3)

    indicator = torch.empty(batch_size, sequence_length, dtype=torch.long, device=device)
    indicator[:, :text_length] = LLM_TOKEN_INDICATOR
    indicator[:, text_length:] = OUTPUT_IMAGE_INDICATOR

    attn_mask = None
    if attention_mask is not None:
        segment_ids = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)
        padding = attention_mask == 0
        segment_ids[:, :text_length][padding] = -1
        indicator[:, :text_length][padding] = 0
        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
    return context, packed, position_ids, attn_mask, indicator, text_length


def _backbone(
    inner: Any,
    llm_features: torch.Tensor | None,
    x: torch.Tensor,
    t: torch.Tensor,
    position_ids: torch.Tensor,
    attn_mask: torch.Tensor | None,
    indicator: torch.Tensor,
    transformer_options: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    indicator = indicator.to(torch.long)
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

    x = x * output_image_mask
    h = inner.input_proj(x) * output_image_mask

    t_cond = inner.t_embedding(t, dtype=x.dtype)
    if t.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(inner.adaln_proj(t_cond))

    if llm_features is not None:
        text_length = llm_features.shape[1]
        text_mask = (indicator[:, :text_length] == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        llm = inner.llm_cond_norm(llm_features * text_mask)
        llm = inner.llm_cond_proj(llm) * text_mask
        h[:, :text_length] = h[:, :text_length] + llm

    h = h + inner.embed_image_indicator(
        (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long), out_dtype=h.dtype
    )

    freqs_cis = precompute_freqs_cis(
        inner.head_dim,
        position_ids[0].transpose(0, 1),
        inner.rope_theta,
        rope_dims=inner.mrope_section,
        interleaved_mrope=True,
        device=position_ids.device,
    )
    freqs_cis = _split_half_rope_matrix(freqs_cis)

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = torch.zeros_like(attn_mask, dtype=h.dtype).masked_fill_(
            ~attn_mask, -torch.finfo(h.dtype).max
        )

    for layer in inner.layers:
        h = layer(h, attn_mask, freqs_cis, adaln_input, transformer_options=transformer_options)
    return h, adaln_input


def _sanitize_prediction(feature: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
    if not dtype.is_floating_point:
        return None
    feature_fp32 = feature.to(torch.float32)
    finite = torch.isfinite(feature_fp32)
    if not bool(finite.any().item()):
        return None
    limits = torch.finfo(dtype)
    return torch.nan_to_num(
        feature_fp32,
        nan=0.0,
        posinf=limits.max,
        neginf=limits.min,
    ).clamp(min=limits.min, max=limits.max).to(dtype)


def _run_with_spectrum(
    executor,
    inner: Any,
    runtime: SpectrumIdeogram4Runtime,
    run_id: int,
    step_id: int,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    batch_size, _, gh, gw = x.shape
    image_token_count = gh * gw
    labels = _model_branch_labels(inner, transformer_options, batch_size)
    topology = _topology_signature(inner, x, context, attention_mask, image_token_count)
    expected_shape = (batch_size, image_token_count, _feature_width(inner))
    call_id, actual = runtime.begin_model_call(
        run_id,
        step_id,
        topology=topology,
        labels=labels,
        expected_shape=expected_shape,
    )

    t = 1.0 - timesteps
    if not actual:
        predicted = runtime.predict(
            run_id,
            step_id,
            call_id,
            device=x.device,
            dtype=x.dtype,
        )
        if predicted is not None:
            sanitized = _sanitize_prediction(predicted, x.dtype)
            if sanitized is not None:
                t_cond = inner.t_embedding(t, dtype=x.dtype)
                if t.dim() == 1:
                    t_cond = t_cond.unsqueeze(1)
                adaln_input = F.silu(inner.adaln_proj(t_cond))
                output = inner.final_layer(sanitized, adaln_input)
                return -inner._tokens_to_img(output, gh, gw)
            runtime.fallback_current_step(run_id, step_id, "forecast feature sanitization produced no finite values")

    llm_features, packed, position_ids, attn_mask, indicator, text_length = _packed_inputs(
        inner, x, context, attention_mask, gh, gw
    )
    hidden, adaln_input = _backbone(
        inner,
        llm_features,
        packed,
        t,
        position_ids,
        attn_mask,
        indicator,
        transformer_options,
    )
    feature = hidden[:, text_length:]
    runtime.observe_actual(run_id, step_id, call_id, feature)
    output = inner.final_layer(hidden, adaln_input)
    return -inner._tokens_to_img(output[:, text_length:], gh, gw)


def diffusion_model_wrapper(
    executor,
    x,
    timesteps,
    context=None,
    attention_mask=None,
    transformer_options=None,
    **kwargs,
):
    options = transformer_options or {}
    runtime = options.get(RUNTIME_KEY)
    run_id = options.get(RUN_ID_KEY)
    step_id = options.get(STEP_ID_KEY)
    if not isinstance(runtime, SpectrumIdeogram4Runtime) or run_id is None or step_id is None:
        return executor(x, timesteps, context, attention_mask, options, **kwargs)

    inner = executor.class_obj
    if not is_native_ideogram4(inner):
        runtime.fallback_current_step(int(run_id), int(step_id), "diffusion model is not native Ideogram4")
        return executor(x, timesteps, context, attention_mask, options, **kwargs)
    if not torch.is_tensor(x) or x.ndim != 4 or not torch.is_tensor(timesteps):
        runtime.fallback_current_step(int(run_id), int(step_id), "Ideogram4 input tensors are not in the native layout")
        return executor(x, timesteps, context, attention_mask, options, **kwargs)
    if context is not None and not torch.is_tensor(context):
        runtime.fallback_current_step(int(run_id), int(step_id), "Ideogram4 context is not a tensor")
        return executor(x, timesteps, context, attention_mask, options, **kwargs)

    return _run_with_spectrum(
        executor,
        inner,
        runtime,
        int(run_id),
        int(step_id),
        x,
        timesteps,
        context,
        attention_mask,
        options,
    )


def install_ideogram4_wrapper(model: Any) -> None:
    import comfy.patcher_extension

    require_native_ideogram4(model)
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    if not model.get_wrappers(wrapper_type, "spectrum_ideogram4"):
        model.add_wrapper_with_key(wrapper_type, "spectrum_ideogram4", diffusion_model_wrapper)
