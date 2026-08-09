from __future__ import annotations

import logging

if __package__:
    from .comfyui_spectrum_ideogram4.config import SpectrumIdeogram4Config
    from .comfyui_spectrum_ideogram4.ideogram4 import (
        install_ideogram4_wrapper,
        require_native_ideogram4,
    )
    from .comfyui_spectrum_ideogram4.runtime import SpectrumIdeogram4Runtime
    from .comfyui_spectrum_ideogram4.sampling import install_sampler_wrappers
else:
    from comfyui_spectrum_ideogram4.config import SpectrumIdeogram4Config
    from comfyui_spectrum_ideogram4.ideogram4 import (
        install_ideogram4_wrapper,
        require_native_ideogram4,
    )
    from comfyui_spectrum_ideogram4.runtime import SpectrumIdeogram4Runtime
    from comfyui_spectrum_ideogram4.sampling import install_sampler_wrappers

LOG = logging.getLogger(__name__)


def _effective_bootstrap_first_forecast(*, requested: bool, degree: int, warmup_steps: int) -> bool:
    if not requested or (degree == 1 and warmup_steps <= 1):
        return requested
    LOG.warning(
        "Spectrum Ideogram4: bootstrap_first_forecast requires degree=1 and warmup_steps<=1; "
        "disabling it for this execution."
    )
    return False


class SpectrumApplyIdeogram4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Chebyshev degree used after the actual history is ready.",
                    },
                ),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 15, "min": 0, "max": 64, "step": 1}),
                "tail_actual_steps": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "max_history": ("INT", {"default": 15, "min": 2, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "history_storage": (["system_ram", "vram"], {"default": "system_ram"}),
                "bootstrap_first_forecast": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use a one-point hold for solver step 1. Requires degree=1 and warmup_steps<=1.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(
        self,
        model,
        enabled,
        blend_weight,
        degree,
        ridge_lambda,
        window_size,
        flex_window,
        warmup_steps,
        tail_actual_steps,
        max_history,
        debug,
        history_storage="system_ram",
        bootstrap_first_forecast=False,
    ):
        if not enabled:
            return (model,)

        require_native_ideogram4(model)
        resolved_degree = int(degree)
        resolved_warmup = int(warmup_steps)
        effective_bootstrap = _effective_bootstrap_first_forecast(
            requested=bool(bootstrap_first_forecast),
            degree=resolved_degree,
            warmup_steps=resolved_warmup,
        )
        config = SpectrumIdeogram4Config(
            enabled=True,
            blend_weight=float(blend_weight),
            degree=resolved_degree,
            ridge_lambda=float(ridge_lambda),
            window_size=float(window_size),
            flex_window=float(flex_window),
            warmup_steps=resolved_warmup,
            tail_actual_steps=int(tail_actual_steps),
            max_history=int(max_history),
            history_storage=str(history_storage),
            debug=bool(debug),
            bootstrap_first_forecast=effective_bootstrap,
        ).validate()

        patched = model.clone()
        require_native_ideogram4(patched)
        runtime = SpectrumIdeogram4Runtime(config)
        install_sampler_wrappers(patched, runtime)
        install_ideogram4_wrapper(patched)
        return (patched,)


NODE_CLASS_MAPPINGS = {"SpectrumApplyIdeogram4": SpectrumApplyIdeogram4}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumApplyIdeogram4": "Spectrum Apply Ideogram 4"}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SpectrumApplyIdeogram4",
]
