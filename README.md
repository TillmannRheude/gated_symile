# Hidden in the Multiplicative Interaction

[![arXiv](https://img.shields.io/badge/arXiv-2512.22991-b31b1b.svg)](https://arxiv.org/pdf/2512.22991)

## Abstract
<div align="justify">
Contrastive learning, popularized by CLIP, provides a scalable way to learn se- mantically meaningful representations from paired modalities. However, many real-world domains benefit from joint reasoning over more than two modalities. Symile addresses this setting with a multiplicative interaction objective that pro- motes higher-order cross-modal dependence. Yet, we find that Symile implicitly assumes that all modalities are equally reliable. In practice, modalities beyond image-text pairs can be noisy, corrupted, or less informative, and treating them uniformly can silently degrade performance. This fragility can be hidden in the multiplicative interaction: Symile may outperform CLIP on average even if a single unreliable modality silently corrupts the product terms. We propose Gated Symile, a contrastive gating mechanism that adapts modality contributions on an attention- based, per-candidate basis. The gate suppresses unreliable inputs by interpolating embeddings toward learnable neutral directions and incorporating an explicit NULL option for jointly uninformative evidence. Across a controlled synthetic benchmark that uncovers corruption fragility and two real-world trimodal retrieval datasets where such failures could be masked by averages, Gated Symile achieves higher top-1 retrieval accuracy than ungated Symile and CLIP. More broadly, our results highlight gating as a practical step toward robust multimodal contrastive learning under imperfect and more than two modalities.
</div>

## Get Started
#### Running a Method on Synthetic-XNOR
To get started quickly, you can use the Synthetic-XNOR dataset as described in the following.
Use the `config_synthetic` inside the `main.py`/`main.ipynb` file. Following this config file, adjust paths and parameters and then start training with the main file.

Note: Weights & Biases (WandB) logging is enabled by default and can be managed via the `wandb` section in the config file. If you prefer to disable WandB, you can comment out the related lines in `main.py` or `main.ipynb`.

## Add further datasets, methods, ...
TODO 

## Citation
In case you find our work helpful, we're happy if you cite us as following
```bibtex
@unpublished{Rheude_Eils_Wild_2025, 
  title={...}, 
  url={...}
}