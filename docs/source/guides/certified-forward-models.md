# Certified forward models

diffpes certification is a scientific-assurance record for one execution of a
forward model. It records the model, semantic conventions, normalized inputs,
ordered transformations, validity checks, derivative evidence, information
losses, and policy outcome that accompany an observable.

Certification does **not** assert that a model is universally correct or that
an input artifact is truthful. A statement such as "validated under
`org.diffpes.policy.research.v1`" is bounded by the recorded model domain,
checks, evidence, and numerical tolerances.

## The JAX-native assurance path

The scientific part of certification executes through the same JAX program as
the ordinary forward model. Contract predicates are array functions, domain
margins and residuals remain differentiable leaves, and information-flow
evidence is computed with JVPs and VJPs. The discrete pass/fail fields are views
of those continuous quantities.

This separation matters during optimization. A policy may report that a point
is outside a verified photon-energy domain, while the algebraic domain margin
still supplies a useful gradient back toward the domain. Likewise, a small VJP
reports local first-order insensitivity at the evaluation point; it is not
treated as proof of global independence.

Filesystem discovery and persistence sit at the JAX-to-world boundary. They
cannot satisfy physics, numerical, or differentiability claims.

## What a certificate records

A {class}`~diffpes.types.ForwardCertificate` contains:

- a versioned forward-model specification and execution manifest;
- source and normalized-artifact references;
- semantic conventions and validity-domain predicates;
- an ordered provenance graph with preserved, introduced, and destroyed
  information;
- claims with continuous margins and references to supporting evidence;
- derivative, structural-dependency, local-sensitivity, and information-spectrum
  diagnostics;
- the evaluated policy report and any unmet requirements.

The graphs answer different questions and are intentionally kept distinct.
Provenance records how a result was produced. Structural dependency records
which input leaves occur in the traced program. JVP/VJP evidence records local
first-order information flow. Transformation records say what interpretation
was preserved or lost, such as absolute intensity calibration after
normalization or high-frequency structure after convolution.

## Certification levels

Levels are recomputed from claims under a named policy. They are not a stored
`certified=True` flag.

| Level | Bounded meaning |
| --- | --- |
| `identified` | Model, inputs, conventions, environment, and output are identified. |
| `validated` | Required contracts and in-domain runtime checks passed. |
| `differentiable` | Declared derivatives were evaluated and met their evidence tolerances. |
| `verified` | Required analytic or external-reference comparisons passed. |
| `benchmarked` | A named independent model or dataset comparison passed in its declared domain. |
| `reproducible` | Resolver-backed re-execution met the recorded reproduction tolerance. |

`failed`, `not_checked`, `not_applicable`, and `out_of_domain` remain visible on
individual claims. A waiver explains an unmet policy requirement; it never
turns the corresponding claim into a pass.

## Inspection

The inspection API is designed to answer scientific questions without loading
the result arrays:

```python
from diffpes.certify import (
    diff_certificates,
    explain_claim,
    summarize_certificate,
)

print(summarize_certificate(certificate))
print(explain_claim(certificate, "claim.output.finite"))
comparison = diff_certificates(reference, candidate)
```

`diff_certificates` separates scientific model, input, transformation,
evidence, numerical, environment, and audit-time differences. This makes a
timestamp-only rerun visibly different from a model or convention change.

## Portable records

Canonical JSON is the authoritative portable representation. Numeric arrays
are stored losslessly with dtype and shape metadata; they are not rounded into
decimal lists. HDF5 stores the exact JSON record and a small set of convenience
attributes next to a numerical result:

```python
from diffpes.inout import (
    attach_certificate_h5,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)

save_certificate_json(certificate, "run.certificate.json")
restored = load_certificate_json("run.certificate.json")

attach_certificate_h5("spectrum.h5", "spectrum", certificate)
restored_from_h5 = load_certificate_h5("spectrum.h5", "spectrum")
```

The record includes a consistency marker that detects accidental storage
mismatches. This bookkeeping value provides no security, authenticity, or
physical-validity claim. Unknown schema major versions are rejected. Unknown
minor extension data is retained in the certificate's extension object so a
load/save cycle does not silently discard it.

## Reading claims responsibly

A certificate can establish that the declared implementation ran with the
recorded inputs and met named tests in a declared domain. It cannot determine
whether the chosen approximation is suitable for an unrecorded experiment,
prove that an upstream DFT or instrument file is truthful, or turn local
Jacobian rank into global identifiability. Those conclusions require upstream
evidence and scientific judgment.
