# Hidden in the Multiplicative Interaction

[Paper link removed for anonymous review](https://anonymous.invalid/paper)

## Abstract
<div align="justify">
Multimodal contrastive learning is increasingly enriched by going beyond image-text pairs. Among recent contrastive methods, Symile is a strong approach for this challenge because its multiplicative interaction objective captures higher-order cross-modal dependence. Yet, we find that Symile treats all modalities symmetrically and does not explicitly model reliability differences, a limitation that becomes especially present in trimodal multiplicative interactions. In practice, modalities beyond image-text pairs can be misaligned, weakly informative, or missing, and treating them uniformly can silently degrade performance. This fragility can be hidden in the multiplicative interaction: Symile may outperform pairwise CLIP even if a single unreliable modality silently corrupts the product terms. We propose Gated Symile, a contrastive gating mechanism that adapts modality contributions on an attention-based, per-candidate basis. The gate suppresses unreliable inputs by interpolating embeddings toward learnable neutral directions and incorporating an explicit NULL option when reliable cross-modal alignment is unlikely. Across a controlled synthetic benchmark that uncovers this fragility and three real-world trimodal datasets for which such failures could be masked by averages, Gated Symile achieves higher top-1 retrieval accuracy than ungated Symile and CLIP. More broadly, our results highlight gating as a step toward robust multimodal contrastive learning under imperfect and more than two modalities.
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