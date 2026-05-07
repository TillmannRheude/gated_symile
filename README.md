# Hidden in the Multiplicative Interaction

[Paper link removed for anonymous review](https://anonymous.invalid/paper)

## Abstract
<div align="justify">
Contrastive learning has become a standard approach for unsupervised learning from paired data, as demonstrated by CLIP for image-text matching. However, many domains involve more than two modalities and require objectives that capture higher-order dependencies beyond pairwise alignment. Symile extends CLIP to this setting by replacing the dot product with the multilinear inner product (MIP) over modality embeddings. In this work, we show that there is a fragility which ishidden in the multiplicative interaction: a single weakly informative, misaligned, or missing modality can propagate through the objective and distort cross-modal retrieval scores. We propose Gated Symile, a contrastive gating mechanism that adapts modality contributions on an attention-based, per-candidate basis. The gate suppresses unreliable inputs by interpolating embeddings toward learnable neutral directions with an explicit NULL option when reliable cross-modal alignment is unlikely. Across a controlled synthetic benchmark that uncovers this fragility and three real-world trimodal datasets, Gated Symile achieves higher top-1 retrieval accuracy than well-tuned state-of-the-art (sota) baselines. More broadly, our results highlight gating as a step toward robust multimodal contrastive learning beyond two modalities in the presence of noise, misalignment, or missing inputs.
</div>

## Get Started
#### Running a Method on Synthetic-XNOR
To get started quickly, you can use the Synthetic-XNOR dataset as described in the following.
Use the `config_synthetic` inside the `main.py`/`main.ipynb` file. Following this config file, adjust paths and parameters and then start training with the main file. For example, you can choose the model (clip or symile) via modelname in `config_synthetic`. The gate can be activated with the symile configuration file under `config/modelname/symile.yaml`. The swap value (p) and other parameters for the synthetic dataset can be adjusted under `config/encoders/synthetic_xnor.yaml`. 

Note: Weights & Biases (WandB) logging is enabled by default and can be managed via the `wandb` section in the config file. If you prefer to disable WandB, you can comment out the related lines in `main.py` or `main.ipynb`.

## Add further datasets, methods, ...
TODO 

## Citation
In case you find our work helpful, we're happy if you cite us as following
```bibtex
@unpublished{Anonymous_2025, 
  title={Anonymous submission title}, 
  url={https://anonymous.invalid/paper}
}