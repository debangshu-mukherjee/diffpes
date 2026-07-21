# Inspect and persist a certified forward run

This tutorial starts after a call to
{func}`diffpes.certify.certify_forward`, which returns a
{class}`~diffpes.types.CertifiedResult`. Its `value` is the ordinary JAX
forward result; its `certificate` is the scientific-assurance record.

```python
from diffpes.certify import (
    certify_forward,
    explain_claim,
    prepare_certification,
    register_builtin_models,
    summarize_certificate,
)
from diffpes.types import (
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    make_execution_manifest,
)

register_builtin_models()

manifest = make_execution_manifest(
    execution_id="tb-radial-example",
    model_ref=f"{TB_RADIAL_MODEL_ID}@{TB_RADIAL_MODEL_VERSION}",
    schema_version="1",
    package_version="development",
    source_checksum="working-tree",
    environment_checksum="tutorial-environment",
    backend="cpu",
    precision_policy="float64",
    deterministic=True,
    started_at_utc="2026-07-21T00:00:00Z",
)

context = prepare_certification(
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    manifest,
    policy_id="org.diffpes.policy.research.v1",
)

# A single PyTree: bands, radial parameters, simulation parameters,
# polarization, work function, optional self-energy, radial grid, and dk.
model_inputs = (
    bands,
    radial,
    simulation,
    polarization,
    4.5,
    None,
    radial_grid,
    None,
)
run = certify_forward(context, model_inputs)
```

The forward model and its scientific checks run together through JAX. The
envelope does not change the spectrum, JVP, or VJP produced by the ordinary
model path.

## Read the record

Start with the stable text summary rather than inspecting the PyTree fields by
hand:

```python
print(summarize_certificate(run.certificate))
```

The summary identifies the model and implementation, policy outcome, input
artifacts, assumptions and conventions, transformations and information
losses, claims by status, and derivative/information diagnostics.

To see why a particular claim passed or failed, request it by its stable ID:

```python
print(
    explain_claim(
        run.certificate,
        "claim.output.finite",
    )
)
```

The explanation includes the subject, continuous margin, evidence method,
reference values, residuals, and tolerances. A zero local sensitivity is
reported as a local observation, never as global independence.

## Save beside the result

Use JSON when the certificate travels independently:

```python
from diffpes.inout import save_certificate_json

save_certificate_json(run.certificate, "tb-radial.certificate.json")
```

Use HDF5 attachment when the numerical result is already stored in an HDF5
file:

```python
from diffpes.inout import attach_certificate_h5, save_to_h5

save_to_h5("tb-radial.h5", spectrum=run.value)
attach_certificate_h5("tb-radial.h5", "spectrum", run.certificate)
```

The HDF5 attachment contains the exact authoritative JSON bytes. Both forms
load to the same {class}`~diffpes.types.ForwardCertificate`:

```python
from diffpes.inout import load_certificate_h5, load_certificate_json

from_json = load_certificate_json("tb-radial.certificate.json")
from_h5 = load_certificate_h5("tb-radial.h5", "spectrum")
```

The stored consistency marker detects accidental storage mismatches only. It
provides no security or physical-assurance claim. Scientific assurance comes
from the explicit contracts, differentiable diagnostics, evidence, and policy
evaluation recorded in the certificate.

## Compare reruns

```python
from diffpes.certify import diff_certificates

change = diff_certificates(from_json, later_run.certificate)
print(change.summary)
```

The comparison distinguishes model/input/semantic changes from numerical,
environment, and audit-only differences. Review scientific differences before
comparing spectra: identical array shapes do not imply identical physical
meaning.
