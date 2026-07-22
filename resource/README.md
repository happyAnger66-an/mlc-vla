# Algorithm team delivery guide -- minimal inference-only runnable model

This guide tells algorithm teams how to organise and submit a model for integration: a
flat, **inference-only**, **runnable** package that reproduces golden outputs. Each model
is delivered under its own directory, `resource/<model_name>/`.

`resource/plannn3/` is the worked reference example -- read it alongside this guide. The
formal contract is `docs/design/algorithm_team_delivery_contract.md`.

## Principles

- **Inference-only.** Include only code reachable from your inference entry points. No
  training/loss modules, optimisers, or unused encoders/heads/classes.
- **Minimal.** Every committed file must be needed to construct and run inference -- if a
  file (or class) is never reached on the inference path, it does not belong here.
- **Runnable and self-validating.** `infer.py` loads the model, runs the real inference
  path, and checks the outputs against golden data; it exits non-zero on mismatch.
- **No large files in git.** Weights and golden input/output pickles stay on NFS;
  `data.json` points to them. Never commit weights, pickles, or other binaries.
- **Self-contained imports.** Code imports resolve inside your `model/` package. Run
  `infer.py` from its own directory so the package is importable without path hacks.

## Required structure

```
resource/<model_name>/
|-- data.json          # manifest: NFS paths to weights + golden input/output
|-- model_config.json  # architecture config (dynamic type strings resolve inside model/)
|-- io_spec.json       # machine-readable input/output names, shapes, dtypes
|-- infer.py           # runnable inference + golden validation
`-- model/             # minimal inference-only model code
    ` ...              # your modules; subpackages (encoder/, head/, ...) as needed
```

Helper scripts for preparing delivery artifacts (such as `prepare_model_config.py`) live
under `scripts/<model_name>/` at the repository root, not under `resource/<model_name>/`.

## Submission checklist

- [ ] `data.json` with the required fields (below) and valid NFS paths.
- [ ] `model/` containing only inference-reachable code (no training/loss/unused code).
- [ ] Run `python3 scripts/<model_name>/prepare_model_config.py <config.json> resource/<model_name>/model_config.json` (see below) and commit the result.
- [ ] `io_spec.json` listing every input and output (name, shape, dtype).
- [ ] `infer.py` that builds the model, runs the real inference path, validates against
      golden, and exits non-zero on mismatch.
- [ ] No weights, golden pickles, or other large/binary files committed.
- [ ] `python3 resource/<model_name>/infer.py` prints PASS and exits 0.

## data.json schema

Required fields:

```json
{
  "model": "<model_name>",
  "version": "v1.0.0",
  "raw_model_defination": "<NFS path to your raw model code>",
  "weights_path": "<NFS path to the checkpoint .bin>",
  "golden_inputs_path": "<NFS path to inputs pickle>",
  "golden_outputs_path": "<NFS path to outputs pickle>"
}
```

Optional: `updated`, `contact`, `notes`. Golden inputs are an ordered list of arrays
(matching your inference entry point's argument order); golden outputs are a dict (or
list) of reference tensors.

## io_spec.json

Machine-readable, not prose. List inputs and outputs in order, each with `name`, `shape`
and `dtype`; optionally describe the inference `stages`. `infer.py` should read the input
names from here so there is a single source of truth. See `resource/plannn3/io_spec.json`.

## infer.py

Structure it so the model-specific logic is reusable:

- `build_model(config_path, weights_path)` -- construct the model and load weights.
- `run_inference(model, inputs) -> dict[str, Tensor]` -- the real inference path, which
  may be multi-stage or autoregressive (not necessarily a single `forward()`).
- validation -- compare each output against golden: exact for discrete outputs; a small
  relative tolerance is acceptable for continuous tensors (e.g. under TF32). Print a
  per-output verdict and exit non-zero on mismatch.

Load golden data with the shared loader:
`from lasermodel_exporter.loaders import load_golden_data`.

A future unified runner (AI Infra, a later roadmap step) will reuse these functions to
drive any model generically, so keeping `build_model` / `run_inference` cleanly separated
now makes that integration free.

## Preparing `model_config.json` from `config.json`

Your original training `config.json` likely contains training-only fields (`loss` entries,
package-prefixed `type` strings, etc.) that should not be in the inference package. Do
**not** copy it as-is -- instead run the model-specific script under `scripts/<model_name>/`:

```bash
python3 scripts/<model_name>/prepare_model_config.py \
    <path/to/algorithm/config.json> \
    resource/<model_name>/model_config.json
```

The script performs delivery-specific transformations (for example):
- Strip algorithm package prefixes from `type` strings so they resolve inside `model/`
- Remove `loss` (never used in inference) and null `head` entries
- Keep non-null `head`s (e.g. `traj_head`) that are part of the inference path
- Persist other architectural fields unchanged

Each `scripts/<model_name>/` owns its own `prepare_model_config.py` script because
schema differences are delivery-specific. The canonical `build_from_model_config` in `src/`
never hard-codes any delivery's field names.

**Do not hand-edit `model_config.json`** -- regenerate it from the original `config.json`
so the transformation is reproducible and diff-able.

## Reference example: plannn3

`resource/plannn3/` is a full worked example. Its inference is genuinely multi-stage
(`trace_encoder_forward -> prefill -> 18x decode`, not a single `forward()`), it keeps
the standard libraries (transformers, timm), and on validation its discrete output
(`traj_ids`) reproduces golden bit-exactly. See `resource/plannn3/README.md` for the
model-specific details and the extraction notes.
