# Evaluation Package

This package contains experiment and evaluation scaffolding for the OPTI-MED ML pivot.

The intended first model family is interpretable tabular models.

That preference is intentional for early work because:

- the feature space is structured and provenance-heavy
- the benchmark is rule-based and inspectable
- the team needs clear comparisons and failure analysis early

Even so, this package must remain model-agnostic.

It should define:

- split contracts
- leakage-prevention rules
- ranking and calibration evaluation interfaces
- baseline comparison contracts
- clinician-review export interfaces

It should not hard-code any specific learner family into the evaluation interfaces.
