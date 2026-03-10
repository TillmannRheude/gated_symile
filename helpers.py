import random
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
import copy 

# Datamodules 
from lightningdatamodules.symile_mimic import DataModule_SymileMimic
from lightningdatamodules.symile_m3 import DataModule_SymileM3
from lightningdatamodules.synthetic_xor import DataModule_SyntheticXOR

# Lightningmodules 
from lightningmodules.symile_mimic import SymileMIMICModel
from lightningmodules.symile_m3 import SymileM3Model
from lightningmodules.ukb import UKBModel
from lightningmodules.synthetic_xor import SyntheticXORModel

# Architecture 
from architecture import Contrastive_Model
from encoders import (
    CXREncoder, ECGEncoder, LabsEncoder,  # symile_mimic
    AudioEncoder, ImageEncoder, TextEncoder,  # symile_m3
    UKBTabularEncoder, UKBMRIEncoder, UKBMRIEncoder2D, Naive3DCNNEncoder,  # ukb
    SyntheticXOREncoder  # synthetic_xor
)

def build_model(cfg: dict):
    # General parameters
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
        "gate_num_heads": cfg["modelname"]["gate_num_heads"],
        "gate_d_null": cfg["modelname"]["gate_d_null"],
        "gate_strength_init": cfg["modelname"]["gate_strength_init"],
        "gate_type": cfg["modelname"]["gate_type"],
        "gate_mode": cfg["modelname"]["gate_mode"],
        "neutral_type": cfg["modelname"]["neutral_type"],
        "use_null": cfg["modelname"]["use_null"],
        "renormalize": cfg["modelname"]["renormalize"],
        "pair_num_negatives": cfg["modelname"]["pair_num_negatives"],
        "mimic_retrieval_mode": cfg["encoders"]["mimic_retrieval_mode"] if "mimic_retrieval_mode" in cfg["encoders"] else None,
        "mimic_global_query_chunk_size": cfg["encoders"]["mimic_global_query_chunk_size"] if "mimic_global_query_chunk_size" in cfg["encoders"] else None,
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
            "mri_brain": 225,
            "mri_heart": 200,
            "mri_brain_phenotypes": 2135,
        }
        modalities = cfg["encoders"]["modalities"]
        # filter input_dims_modalities to only include modalities in modalities
        input_dims_modalities = {k: v for k, v in input_dims_modalities.items() if k in modalities}

        n_modalities = len(input_dims_modalities)
        is_rocm = getattr(torch.version, "hip", None) is not None
        encoders = nn.ModuleList([])

        for mod in modalities:
            if "mri" in mod and not "phenotypes" in mod:
                encoders.append(
                    Naive3DCNNEncoder(
                        in_ch=1,
                        base=32,
                        emb_dim=cfg["modelname"]["emb_dim"],
                    )
                    #UKBMRIEncoder(
                    #    input_channels=1,
                    #    input_dim=input_dims_modalities[mod],
                    #    vit_params={
                    #        "patch_size": cfg["encoders"][mod]["vit"]["patch_size"],
                    #        "d_model": cfg["encoders"][mod]["vit"]["d_model"],
                    #        "num_layers": cfg["encoders"][mod]["vit"]["num_layers"],
                    #        "num_heads": cfg["encoders"][mod]["vit"]["num_heads"]
                    #    },
                    #    emb_dim=cfg["modelname"]["emb_dim"],
                    #    last_dim_temporal=cfg["encoders"][mod]["vit"]["last_dim_temporal"],
                    #)
                    #UKBMRIEncoder2D(
                    #    input_channels=1,
                    #    vit_params={
                    #        "patch_size": cfg["encoders"][mod]["vit"]["patch_size"],
                    #        "d_model": cfg["encoders"][mod]["vit"]["d_model"],
                    #        "num_layers": cfg["encoders"][mod]["vit"]["num_layers"],
                    #        "num_heads": cfg["encoders"][mod]["vit"]["num_heads"]
                    #    },
                    #    emb_dim=cfg["modelname"]["emb_dim"],
                    #    last_dim_temporal=cfg["encoders"][mod]["vit"]["last_dim_temporal"],
                    #)
                )
                #if same_state_dict is None:
                #    same_state_dict = encoders[-1].state_dict()
                #else:
                #    encoders[-1].load_state_dict(same_state_dict)
                #    allowed = ("transformer.", "projection_head.")
                #    to_load = {}
                #    tgt_sd = encoders[-1].state_dict()
                #    for k, v in same_state_dict.items():
                #        if k == "cls_token" or k.startswith(allowed):
                #            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                #                to_load[k] = v
                #    encoders[-1].load_state_dict(to_load, strict=False)

                # convert last encoder to channels last 3d for faster training, but if cuda
                #encoders[-1] = encoders[-1].to(memory_format=torch.channels_last_3d) if not is_rocm else encoders[-1] 
                #encoders[-1] = encoders[-1].to(memory_format=torch.channels_last) if not is_rocm else encoders[-1]
            else:
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
                #if same_state_dict is not None:
                    # check if "mri" in any mod of modalities
                    #if any("mri" in mod for mod in modalities):
                    #    encoders[-1] = UKBTabularEncoder(
                    #        input_dim=input_dims_modalities[mod],
                    #        hidden_dims=cfg["encoders"][mod]["mlp"]["hidden_dims"],
                    #        hidden_dropouts=cfg["encoders"][mod]["mlp"]["hidden_dropouts"],
                    #        emb_dim=cfg["modelname"]["emb_dim"],
                    #        combine_eids_as=cfg["datamodule"]["combine_eids_as"],
                    #    )
                    #    print("Added adapter to the tabular encoder")
                        
                        #layers_to_add = nn.Sequential(
                        #    nn.Linear(cfg["modelname"]["emb_dim"], cfg["encoders"]["mri_brain"]["vit"]["d_model"]),
                        #    nn.ReLU(),
                        #    nn.Linear(cfg["encoders"]["mri_brain"]["vit"]["d_model"], cfg["modelname"]["emb_dim"]),
                        #)
                        # init weights of the layers_to_add with encoders[-1]._init_weights
                        #layers_to_add.apply(encoders[-1]._init_weights)
                        # copy params from encoders[-2].projection_head to layers_to_add[-1]
                        #layers_to_add[-1].weight.data = copy.deepcopy(encoders[-2].projection_head.weight.data)
                        #layers_to_add[-1].bias.data = copy.deepcopy(encoders[-2].projection_head.bias.data)
                        #encoders[-1].mlp.append(layers_to_add)
                        #print("Added linear layers to the tabular mlp and initialized weights with the projection head of the MRI encoder")

        model = Contrastive_Model(encoders=encoders)

        return UKBModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            candidate_idx=cfg["encoders"]["candidate_idx"],
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            modalities=cfg["encoders"]["modalities"],
        )
    

    # Synthetic XOR
    if cfg["dataset_name"] == "synthetic_xor":
        encoders = nn.ModuleList([
            SyntheticXOREncoder(
                input_dim=cfg["encoders"]["input_dim"], 
                emb_dim=cfg["modelname"]["emb_dim"]
            ),
            SyntheticXOREncoder(
                input_dim=cfg["encoders"]["input_dim"], 
                emb_dim=cfg["modelname"]["emb_dim"]
            ),
            SyntheticXOREncoder(
                input_dim=cfg["encoders"]["input_dim"], 
                emb_dim=cfg["modelname"]["emb_dim"]
            ),
        ])
        model = Contrastive_Model(encoders=encoders)
        return SyntheticXORModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
        )

    # MSR-VTT (TODO)
    if cfg["dataset_name"] == "msr_vtt":
        raise NotImplementedError("MSR-VTT not implemented, yet.")
    
    else:
        raise ValueError(f"Model {cfg['dataset_name']} not implemented.")



def build_datamodule(cfg: dict):
    # Symile-MIMIC
    if cfg["dataset_name"] == "symile_mimic":
        enc_cfg = cfg.get("encoders", {})
        return DataModule_SymileMimic(
            batch_size = cfg["batch_size"],
            split_nr = cfg["split_nr"],
            permute = cfg["encoders"]["permute"],
            permute_p = cfg["encoders"]["permute_p"],
            permute_mod = cfg["encoders"]["permute_mod"],
            permute_seed = cfg["encoders"]["permute_seed"],
        )
    
    # Symile-M3
    if cfg["dataset_name"] == "symile_m3":
        return DataModule_SymileM3(
            batch_size = cfg["batch_size"],
            split_nr = cfg["split_nr"],
            #text_model_id = cfg["text_model_id"],
            #num_langs = cfg["num_langs"],
        )

    # UKBB (TODO)
    if cfg["dataset_name"] == "ukb":
        # lazy import if private UDM not installed 
        from udm.general_datamodule import GeneralDatamodule
        datamodule_params = {key: value for key, value in cfg.datamodule.items()}
        return GeneralDatamodule(**datamodule_params)

    # Synthetic XOR
    if cfg["dataset_name"] == "synthetic_xor":
        return DataModule_SyntheticXOR(
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

    # MSR-VTT (TODO)
    if cfg["dataset_name"] == "msr_vtt":
        raise NotImplementedError("MSR-VTT not implemented, yet.")
    
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
