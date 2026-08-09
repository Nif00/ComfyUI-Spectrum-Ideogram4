# Spectrum for Ideogram 4

![Spectrum for Ideogram 4](SpectrumIdeogram4.png)

This custom node adds Spectrum feature forecasting to ComfyUI's native
Ideogram 4 transformer. It forecasts image-token features between selected
actual evaluations while retaining the native Ideogram output head and falling
back to an actual evaluation when a forecast is not safe.

## Installation

From the ComfyUI installation directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nif00/ComfyUI-Spectrum-Ideogram4.git
```

Restart ComfyUI after installation.

## Workflow placement

For the two-model Ideogram workflow, apply Spectrum to both models with the
same settings:

```text
regular Ideogram4 model        -> ModelSamplingAuraFlow -> Spectrum Apply Ideogram 4 -> DualModelGuider.model
unconditional Ideogram4 model  -> Spectrum Apply Ideogram 4 -> DualModelGuider.model_negative
```

The regular model drives the shared sampler schedule; regular and unconditional
forecast histories remain separate. If CFG is 1, the negative model is skipped.

## Default settings

| Setting | Default |
| --- | ---: |
| Blend weight | `0.50` |
| Degree | `1` |
| Ridge lambda | `0.1` |
| Window size | `2.0` |
| Flex window | `0.75` |
| Warmup steps | `15` |
| Tail actual steps | `2` |
| Max history | `15` |

The node requires ComfyUI's native `Ideogram4Transformer2DModel`. Spectrum step
tracking is supported for Euler, Euler CFG++, RES multistep, and RES multistep
CFG++; other samplers continue through the native path.
