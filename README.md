# Anonymous Review Code

This repository accompanies an anonymous submission on robust multimodal
contrastive learning. It contains the proposed gated multilinear method,
pairwise and geometric baselines, synthetic experiments, and implementations
for the real-world evaluation pipelines described in the submission.

## Quick start: Synthetic-XNOR

The Synthetic-XNOR experiment is self-contained and is the recommended way to
verify the implementation. Select `config/config_synthetic.yaml` in `main.py`
or `main.ipynb`, adjust the training parameters if needed, and run the entry
point. The method is selected in the config defaults; gating options are in
`config/modelname/symile.yaml`, and dataset parameters are in
`config/encoders/synthetic_xnor.yaml`.

Weights & Biases logging is enabled by default. Disable or replace the logger
in `main.py` when running without a WandB account.

## Data-dependent experiments

The real-world pipelines require datasets that cannot be redistributed with
this repository. Paths in `config/datamodule/` are anonymized placeholders and
must be changed to local dataset locations. No private data, credentials, or
trained checkpoints are included.

## Anonymous review

Author, paper, repository, and citation links are intentionally omitted during
double-blind review. Citation information will be restored after review.
