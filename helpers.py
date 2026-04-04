import random
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

# Datamodules 
from lightningdatamodules.symile_mimic import DataModule_SymileMimic
from lightningdatamodules.symile_m3 import DataModule_SymileM3
from lightningdatamodules.synthetic_xnor import DataModule_SyntheticXNOR

# Lightningmodules 
from lightningmodules.symile_mimic import SymileMIMICModel
from lightningmodules.symile_m3 import SymileM3Model
from lightningmodules.ukb import UKBModel
from lightningmodules.synthetic_xnor import SyntheticXNORModel

# Architecture 
from architecture import Contrastive_Model
from architecture import CoMM_Model
from architecture import TransformerSymile_Model
from encoders import (
    CXREncoder, ECGEncoder, LabsEncoder,  # symile_mimic
    AudioEncoder, ImageEncoder, TextEncoder,  # symile_m3
    UKBTabularEncoder, # ukb
    SyntheticXNOREncoder  # synthetic_xnor
)

def build_model(cfg: dict):
    params_optimizer = {
        "name": cfg["optimizer"]["name"],
        "lr": cfg["optimizer"]["lr"],
        "lr_gate_mul": cfg["optimizer"]["lr_gate_mul"],
        "weight_decay": cfg["optimizer"]["weight_decay"],
        "eps": cfg["optimizer"]["eps"],
        "betas": cfg["optimizer"]["betas"],
        "warmup_steps": cfg["optimizer"]["warmup_steps"],
    }
    params_method = {
        "modelname": cfg["modelname"]["modelname"],
        "negative_sampling": cfg["modelname"]["negative_sampling"],
        "logit_scale_init": cfg["modelname"]["logit_scale_init"],
        "batch_size": cfg["batch_size"],
        "embedding_norm": cfg["modelname"]["embedding_norm"],
        "bias_init_mult": cfg["modelname"]["bias_init_mult"] if "bias_init_mult" in cfg["modelname"] else 1.0,
        "use_gate": cfg["modelname"]["use_gate"],
        "gate_temp": cfg["modelname"]["gate_temp"],
        "gate_bias_init": cfg["modelname"]["gate_bias_init"],
        "gate_d_k": cfg["modelname"]["gate_d_k"],
        "gate_strength_init": cfg["modelname"]["gate_strength_init"],
        "gate_type": cfg["modelname"]["gate_type"],
        "gate_mode": cfg["modelname"]["gate_mode"],
        "neutral_type": cfg["modelname"]["neutral_type"],
        "use_null": cfg["modelname"]["use_null"],
        "renormalize": cfg["modelname"]["renormalize"],
        "pair_num_negatives": cfg["modelname"]["pair_num_negatives"],
    }
    params_retrival_ds = {
        "batch_size": cfg["batch_size"],
        "split_nr": cfg["split_nr"],
    }
    modelname = cfg["modelname"]["modelname"]

    # Symile-MIMIC
    if cfg["dataset_name"] == "symile_mimic":
        encoders = nn.ModuleList([
            CXREncoder(emb_dim=cfg["modelname"]["emb_dim"]),
            ECGEncoder(emb_dim=cfg["modelname"]["emb_dim"]),
            LabsEncoder(emb_dim=cfg["modelname"]["emb_dim"]),
        ])
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
            )

        return SymileMIMICModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
        )

    # Symile-M3
    if cfg["dataset_name"] == "symile_m3":
        encoders = nn.ModuleList([
            AudioEncoder(emb_dim=cfg["modelname"]["emb_dim"]),
            ImageEncoder(emb_dim=cfg["modelname"]["emb_dim"]),
            TextEncoder(emb_dim=cfg["modelname"]["emb_dim"]),
        ])
        model = Contrastive_Model(encoders=encoders)

        return SymileM3Model(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
        )
    
    # UKBB 
    if cfg["dataset_name"] == "ukb":
        input_dims_modalities = {
            "nmr": 249,
            "ehr": 3584,
            "ehr_imaging": 4096,
            "olink": 1463,
            "prs": 135,
            "bloodbio": 30,
            "baselinechars": 28,
            "localenvironment": 33,
            "arterialstiffness": 9,
            "anthropometry": 43,
            "bloodpressure": 12,
            "ecgduringexercise": 355,
            "eyemeasures": 310,
            "bonedensitometry": 27,
            "handgripstrength": 2,
            "spirometry": 29,
            "touchscreen": 119,
            "cognitivefunction": 27,
            "hearingtest": 68,
            "verbalinterview": 224,
            "bloodcount": 31,
            "urineassays": 4,
            "telomeres": 4,
            "infectiousdiseases": 66,
        }
        modalities = cfg["encoders"]["modalities"]
        input_dims_modalities = {k: v for k, v in input_dims_modalities.items() if k in modalities}
        encoders = nn.ModuleList([])

        for mod in modalities:
            encoders.append(
                UKBTabularEncoder(
                    input_dim=input_dims_modalities[mod],
                    hidden_dims=cfg["encoders"][mod]["mlp"]["hidden_dims"],
                    hidden_dropouts=cfg["encoders"][mod]["mlp"]["hidden_dropouts"],
                    emb_dim=cfg["modelname"]["emb_dim"],
                    combine_eids_as=cfg["datamodule"]["combine_eids_as"],
                    modality_name=mod,
                )
            )
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
            )

        return UKBModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            candidate_idx=cfg["encoders"]["candidate_idx"],
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            modalities=cfg["encoders"]["modalities"],
        )
    
    # Synthetic XNOR
    if cfg["dataset_name"] == "synthetic_xnor":
        encoders = nn.ModuleList([
            SyntheticXNOREncoder(
                input_dim=cfg["encoders"]["input_dim"], 
                emb_dim=cfg["modelname"]["emb_dim"]
            ),
            SyntheticXNOREncoder(
                input_dim=cfg["encoders"]["input_dim"], 
                emb_dim=cfg["modelname"]["emb_dim"]
            ),
            SyntheticXNOREncoder(
                input_dim=cfg["encoders"]["input_dim"], 
                emb_dim=cfg["modelname"]["emb_dim"]
            ),
        ])
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
            )
        if cfg["modelname"]["modelname"] == "comm":
            model = CoMM_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                # augmentation_params=cfg["modelname"]["augmentation_params"],
            )
        return SyntheticXNORModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
        )
    
    else:
        raise ValueError(f"Model {cfg['dataset_name']} not implemented.")



def build_datamodule(cfg: dict):
    # Symile-MIMIC
    if cfg["dataset_name"] == "symile_mimic":
        enc_cfg = cfg.get("encoders", {})
        return DataModule_SymileMimic(
            batch_size = cfg["batch_size"],
            split_nr = cfg["split_nr"],
        )
    
    # Symile-M3
    if cfg["dataset_name"] == "symile_m3":
        return DataModule_SymileM3(
            batch_size = cfg["batch_size"],
            split_nr = cfg["split_nr"],
            #text_model_id = cfg["text_model_id"],
            #num_langs = cfg["num_langs"],
        )

    # UKBB 
    if cfg["dataset_name"] == "ukb":
        # lazy import if private UDM not installed 
        from udm.general_datamodule import GeneralDatamodule
        datamodule_params = {key: value for key, value in cfg.datamodule.items()}
        return GeneralDatamodule(**datamodule_params)

    # Synthetic XOR
    if cfg["dataset_name"] == "synthetic_xnor":
        return DataModule_SyntheticXNOR(
            batch_size = cfg["batch_size"],
            n_samples = cfg["encoders"]["n_samples"],
            dims_modality = cfg["encoders"]["dims_modality"],
            n_bits = cfg["encoders"]["n_bits"],
            embed_mode = cfg["encoders"].get("embed_mode", "xnor_only"),
            bc_corr_exclusive = cfg["encoders"].get("bc_corr_exclusive", False),
            bc_corr_p = cfg["encoders"].get("bc_corr_p", None),
            bc_corr_split = cfg["encoders"].get("bc_corr_split", 0.5),
            p_flips = cfg["encoders"]["p_flips"],
            p_corrs = cfg["encoders"]["p_corrs"],
            corr_modes = cfg["encoders"]["corr_modes"],
            signal_scale = cfg["encoders"]["signal_scale"],
            distractor_std = cfg["encoders"]["distractor_std"],
            a_rule = cfg["encoders"]["a_rule"],
            seed = cfg["seed"],
        )
    
    else:
        raise ValueError(f"Data module {cfg['dataset_name']} not implemented.")



def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    #torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
