# AANA Identity Baseline

This example builds a minimal causal-variable-track submission for an
AANA-style identity/control featurizer. It is intended to exercise MIB's
submission contract before running a full leaderboard evaluation.

The generated submission uses the built-in identity featurizer modules:

- `IdentityFeaturizerModule`
- `IdentityInverseFeaturizerModule`

In AANA terms, this is a verifier-gated control configuration: the featurizer
leaves the activation unchanged, while the generation script writes a manifest
recording the checked constraint surface and the exact MIB submission files.
It should be treated as a format-valid baseline, not a mechanistic discovery
method or leaderboard result.

## Generate and Verify

From `MIB-causal-variable-track`:

```bash
python examples/aana_identity_baseline/generate_submission.py --output-dir aana_identity_submission
python verify_submission.py aana_identity_submission
```

The verification run performed for this example passed MIB's submission
checker with one valid triplet:

```text
Perfect submission! No errors or warnings. Found 1 valid triplet(s).
```

Full benchmark scoring still requires running MIB's model evaluation pipeline
or submitting the generated artifact to the official leaderboard.
