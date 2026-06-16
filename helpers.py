import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
import torch

# Datamodules 
from lightningdatamodules.symile_mimic import DataModule_SymileMimic
from lightningdatamodules.symile_m3 import DataModule_SymileM3
from lightningdatamodules.synthetic_xnor import DataModule_SyntheticXNOR, DataModule_SyntheticXNORBimodal
from lightningdatamodules.synthetic_xor_symile import DataModule_SymileXOR
from lightningdatamodules.mcmed import DataModule_MCMED
from lightningdatamodules.mimic import DataModule_MIMIC

# Lightningmodules 
from lightningmodules.symile_mimic import SymileMIMICModel
from lightningmodules.symile_m3 import SymileM3Model
from lightningmodules.ukb import UKBModel
from lightningmodules.synthetic_xnor import SyntheticXNORModel, SyntheticXNORBimodalModel
from lightningmodules.symile_xor import SymileXORModel
from lightningmodules.mcmed import MCMEDModel
from lightningmodules.mimic import MIMICModel

# Architecture 
from architecture import Contrastive_Model
from architecture import CoMM_Model
from architecture import TransformerSymile_Model
from encoders import (
    CXREncoder, ECGEncoder, LabsEncoder,  # symile_mimic
    AudioEncoder, ImageEncoder, TextEncoder,  # symile_m3
    UKBTabularEncoder, # ukb
    SyntheticXNOREncoder,  # synthetic_xnor
    MCMEDWaveformEncoder, MCMEDRadiologyEncoder, MCMEDNumericsEncoder, MCMEDDemographicsEncoder, # mcmed
    MimicRadBERTEncoder, MimicMLPEncoder, # mimic
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
        "candidate_dependent": cfg["modelname"].get("candidate_dependent", True),
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
        "retrieval_mode": cfg.get("encoders", {}).get("retrieval_mode", "global"),
        "preselected_num_candidates": cfg.get("encoders", {}).get("preselected_num_candidates", 10),
        "retrieval_seed": cfg.get("seed", 420),
    }
    modelname = cfg["modelname"]["modelname"]

    # Symile-MIMIC
    if cfg["dataset_name"] == "symile_mimic":
        encoders = nn.ModuleList([
            CXREncoder(
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"].get("geometry_preserving", False),
            ),
            ECGEncoder(
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"].get("geometry_preserving", False),
            ),
            LabsEncoder(
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"].get("geometry_preserving", False),
                leaky_relu_negative_slope=cfg["encoders"]["leaky_relu_negative_slope"]
            ),
        ])
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
                seq_dims=[1,1,1],
                candidate_dependent=cfg["modelname"].get("candidate_dependent", True),
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
            )

        return SymileMIMICModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            candidate_idx=cfg["encoders"].get("candidate_idx", 1),
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
            "ehr_future": 3062,
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
                    geometry_preserving=cfg["encoders"]["geometry_preserving"],
                    leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
                )
            )
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
                seq_dims=[1,1,1],
                candidate_dependent=cfg["modelname"].get("candidate_dependent", True),
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
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
        n_modalities = int(cfg["encoders"].get("n_modalities", 3))
        bounded_svd_params = cfg["encoders"]["bounded_svd_params"]
        activation_fn_params = cfg["encoders"]["activation_fn_params"]
        feature_layout = cfg["encoders"].get("feature_layout", "1d")
        input_layout = cfg["encoders"].get("input_layout", "1d")
        if feature_layout != input_layout:
            raise ValueError(
                f"For synthetic_xnor, feature_layout ({feature_layout}) and input_layout ({input_layout}) must match."
            )
        if n_modalities == 2:
            if cfg["modelname"]["modelname"] != "clip":
                raise ValueError("Synthetic XNOR bimodal currently supports only modelname='clip'.")
            sequence_encoder_params = cfg["encoders"].get("sequence_encoder_params", {})
            sequence_encoder_params = dict(sequence_encoder_params)
            sequence_encoder_params.setdefault("seq_len", cfg["encoders"].get("seq_len", 4))

            encoders = nn.ModuleList([
                SyntheticXNOREncoder(
                    input_dim=cfg["encoders"]["input_dim"],
                    emb_dim=cfg["modelname"]["emb_dim"],
                    geometry_preserving=cfg["encoders"]["geometry_preserving"],
                    bounded_svd_params=bounded_svd_params,
                    activation_fn_params=activation_fn_params,
                    input_layout=input_layout,
                    sequence_encoder_params=sequence_encoder_params,
                ),
                SyntheticXNOREncoder(
                    input_dim=cfg["encoders"]["input_dim"],
                    emb_dim=cfg["modelname"]["emb_dim"],
                    geometry_preserving=cfg["encoders"]["geometry_preserving"],
                    bounded_svd_params=bounded_svd_params,
                    activation_fn_params=activation_fn_params,
                    input_layout=input_layout,
                    sequence_encoder_params=sequence_encoder_params,
                ),
            ])
            model = Contrastive_Model(encoders=encoders)
            return SyntheticXNORBimodalModel(
                params_optimizer=params_optimizer,
                params_method=params_method,
                model=model,
                modelname=modelname,
                params_retrival_ds=params_retrival_ds,
            )

        if n_modalities == 3:
            sequence_encoder_params = cfg["encoders"].get("sequence_encoder_params", {})
            sequence_encoder_params = dict(sequence_encoder_params)
            sequence_encoder_params.setdefault("seq_len", cfg["encoders"].get("seq_len", 4))
            encoders = nn.ModuleList([
                SyntheticXNOREncoder(
                    input_dim=cfg["encoders"]["input_dim"], 
                    emb_dim=cfg["modelname"]["emb_dim"],
                    geometry_preserving=cfg["encoders"]["geometry_preserving"],
                    bounded_svd_params=bounded_svd_params,
                    activation_fn_params=activation_fn_params,
                    input_layout=input_layout,
                    sequence_encoder_params=sequence_encoder_params,
                ),
                SyntheticXNOREncoder(
                    input_dim=cfg["encoders"]["input_dim"], 
                    emb_dim=cfg["modelname"]["emb_dim"],
                    geometry_preserving=cfg["encoders"]["geometry_preserving"],
                    bounded_svd_params=bounded_svd_params,
                    activation_fn_params=activation_fn_params,
                    input_layout=input_layout,
                    sequence_encoder_params=sequence_encoder_params,
                ),
                SyntheticXNOREncoder(
                    input_dim=cfg["encoders"]["input_dim"], 
                    emb_dim=cfg["modelname"]["emb_dim"],
                    geometry_preserving=cfg["encoders"]["geometry_preserving"],
                    bounded_svd_params=bounded_svd_params,
                    activation_fn_params=activation_fn_params,
                    input_layout=input_layout,
                    sequence_encoder_params=sequence_encoder_params,
                )
            ])
            model = Contrastive_Model(encoders=encoders)
            if cfg["modelname"]["modelname"] == "symile_attention":
                model = TransformerSymile_Model(
                    contrastive_model=model,
                    transformer_params=cfg["modelname"]["transformer_params"],
                    proj_output_dim=1,
                    seq_dims=[1,1,1],
                    candidate_dependent=cfg["modelname"].get("candidate_dependent", True),
                    leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
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
                regularizer_params=cfg["encoders"]["regularizer_params"]
            )

        raise ValueError(f"Unsupported synthetic_xnor n_modalities={n_modalities}. Expected 2 or 3.")

    # Symile XOR (synthetic)
    if cfg["dataset_name"] == "symile_xor":
        bounded_svd_params = cfg["encoders"]["bounded_svd_params"]
        activation_fn_params = cfg["encoders"]["activation_fn_params"]
        input_layout = cfg["encoders"].get("input_layout", "1d")
        if input_layout != "1d":
            raise ValueError("symile_xor currently supports only input_layout='1d'.")

        encoders = nn.ModuleList([
            SyntheticXNOREncoder(
                input_dim=cfg["encoders"]["input_dim"],
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"],
                bounded_svd_params=bounded_svd_params,
                activation_fn_params=activation_fn_params,
                input_layout=input_layout,
                sequence_encoder_params=cfg["encoders"].get("sequence_encoder_params", {}),
            ),
            SyntheticXNOREncoder(
                input_dim=cfg["encoders"]["input_dim"],
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"],
                bounded_svd_params=bounded_svd_params,
                activation_fn_params=activation_fn_params,
                input_layout=input_layout,
                sequence_encoder_params=cfg["encoders"].get("sequence_encoder_params", {}),
            ),
            SyntheticXNOREncoder(
                input_dim=cfg["encoders"]["input_dim"],
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"],
                bounded_svd_params=bounded_svd_params,
                activation_fn_params=activation_fn_params,
                input_layout=input_layout,
                sequence_encoder_params=cfg["encoders"].get("sequence_encoder_params", {}),
            ),
        ])
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
                seq_dims=[1, 1, 1],
                candidate_dependent=cfg["modelname"].get("candidate_dependent", True),
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
            )
        if cfg["modelname"]["modelname"] == "comm":
            model = CoMM_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
            )
        return SymileXORModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            candidate_idx=cfg["encoders"].get("candidate_idx", 1),
        )
    
    
    """ 
    MCMEDWaveformEncoder(
        emb_dim=cfg["modelname"]["emb_dim"],
        d_model=cfg["encoders"]["waveform_ii"]["resnet"]["d_model"],
        num_blocks=cfg["encoders"]["waveform_ii"]["resnet"]["num_blocks"],
        kernel_size=cfg["encoders"]["waveform_ii"]["resnet"]["kernel_size"],
        downsample_stride=cfg["encoders"]["waveform_ii"]["resnet"]["downsample_stride"],
        num_input_proj_layers=cfg["encoders"]["waveform_ii"]["resnet"].get("num_input_proj_layers", 2),
        input_proj_leaky_relu_negative_slope=cfg["encoders"]["waveform_ii"]["resnet"].get("leaky_relu_negative_slope", 0.0),
        dropout=cfg["encoders"]["waveform_ii"]["resnet"]["dropout"],
    ),
    """
    # MC-MED
    if cfg["dataset_name"] == "mcmed":
        encoders = nn.ModuleList([
            MCMEDDemographicsEncoder(
                model_params=cfg["encoders"].get("demographics", {}),
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"]),
            MCMEDRadiologyEncoder(
                model_params=cfg["encoders"].get("radiology", {}),
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"]),
            MCMEDNumericsEncoder(
                emb_dim=cfg["modelname"]["emb_dim"],
                trend_feature_dim=cfg["encoders"]["numerics"].get("trend_feature_dim", 7),
                d_model=cfg["encoders"]["numerics"]["resnet"]["d_model"],
                num_blocks=cfg["encoders"]["numerics"]["resnet"]["num_blocks"],
                kernel_size=cfg["encoders"]["numerics"]["resnet"]["kernel_size"],
                dropout=cfg["encoders"]["numerics"]["resnet"]["dropout"],
            ),
        ])
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
                seq_dims=[1,1,1],
                candidate_dependent=cfg["modelname"].get("candidate_dependent", True),
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
            )
        return MCMEDModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
        )

    # MIMIC (new trimodal variant)
    if cfg["dataset_name"] == "mimic":
        timeseries_input_dim = int(cfg["encoders"]["timeseries"]["input_dim"])
        if timeseries_input_dim <= 0:
            probe_df = pd.read_csv(cfg["encoders"]["train_split_csv"], nrows=1)
            timeseries_input_dim = int(len([c for c in probe_df.columns if c.startswith("ts_item")]))
            if timeseries_input_dim <= 0:
                raise ValueError("Could not infer timeseries input_dim from train_split_csv; please set encoders.timeseries.input_dim.")

        ts_hidden_dims = cfg["encoders"]["timeseries"]["mlp"]["hidden_dims"]
        ts_hidden_dropouts = cfg["encoders"]["timeseries"]["mlp"]["hidden_dropouts"]
        demo_hidden_dims = cfg["encoders"]["demographics"]["mlp"]["hidden_dims"]
        demo_hidden_dropouts = cfg["encoders"]["demographics"]["mlp"]["hidden_dropouts"]
        if len(ts_hidden_dims) != len(ts_hidden_dropouts):
            raise ValueError("mimic.timeseries.mlp.hidden_dims and hidden_dropouts must have the same length.")
        if len(demo_hidden_dims) != len(demo_hidden_dropouts):
            raise ValueError("mimic.demographics.mlp.hidden_dims and hidden_dropouts must have the same length.")

        encoders = nn.ModuleList([
            MimicRadBERTEncoder(
                model_params=cfg["encoders"].get("radiology", {}),
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"],
            ),
            MimicMLPEncoder(
                input_dim=timeseries_input_dim,
                hidden_dims=ts_hidden_dims,
                hidden_dropouts=ts_hidden_dropouts,
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"],
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
            ),
            MimicMLPEncoder(
                input_dim=cfg["encoders"]["demographics"]["input_dim"],
                hidden_dims=demo_hidden_dims,
                hidden_dropouts=demo_hidden_dropouts,
                emb_dim=cfg["modelname"]["emb_dim"],
                geometry_preserving=cfg["encoders"]["geometry_preserving"],
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
            ),
        ])
        model = Contrastive_Model(encoders=encoders)
        if cfg["modelname"]["modelname"] == "symile_attention":
            model = TransformerSymile_Model(
                contrastive_model=model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
                seq_dims=[1, 1, 1],
                candidate_dependent=cfg["modelname"].get("candidate_dependent", True),
                leaky_relu_negative_slope=cfg["encoders"].get("leaky_relu_negative_slope", 0.0),
            )
        return MIMICModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            candidate_idx=cfg["encoders"].get("candidate_idx", 0),
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
        n_modalities = int(cfg["encoders"].get("n_modalities", 3))
        if n_modalities == 2:
            dims_modality = list(cfg["encoders"]["dims_modality"])[:2]
            p_flips = list(cfg["encoders"]["p_flips"])[:1]
            p_corrs = list(cfg["encoders"]["p_corrs"])[:2]
            return DataModule_SyntheticXNORBimodal(
                batch_size=cfg["batch_size"],
                n_samples=cfg["encoders"]["n_samples"],
                dims_modality=dims_modality,
                feature_layout=cfg["encoders"].get("feature_layout", "1d"),
                seq_len=cfg["encoders"].get("seq_len", 4),
                n_bits=cfg["encoders"]["n_bits"],
                embed_mode=cfg["encoders"].get("embed_mode", "xnor_only"),
                ab_corr_exclusive=cfg["encoders"].get("ab_corr_exclusive", False),
                ab_corr_p=cfg["encoders"].get("ab_corr_p", None),
                ab_corr_split=cfg["encoders"].get("ab_corr_split", 0.5),
                p_flips=p_flips,
                p_corrs=p_corrs,
                corr_modes=cfg["encoders"]["corr_modes"],
                signal_scale=cfg["encoders"]["signal_scale"],
                distractor_std=cfg["encoders"]["distractor_std"],
                a_rule=cfg["encoders"]["a_rule"],
                seed=cfg["seed"],
            )

        if n_modalities != 3:
            raise ValueError(f"Unsupported synthetic_xnor n_modalities={n_modalities}. Expected 2 or 3.")

        return DataModule_SyntheticXNOR(
            batch_size = cfg["batch_size"],
            n_samples = cfg["encoders"]["n_samples"],
            dims_modality = cfg["encoders"]["dims_modality"],
            feature_layout = cfg["encoders"].get("feature_layout", "1d"),
            seq_len = cfg["encoders"].get("seq_len", 4),
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

    # Symile XOR (synthetic)
    if cfg["dataset_name"] == "symile_xor":
        return DataModule_SymileXOR(
            batch_size=cfg["batch_size"],
            n_samples=cfg["encoders"]["n_samples"],
            p=cfg["encoders"]["p"],
            value_mode=cfg["encoders"].get("value_mode", "zero_one"),
            dim=cfg["encoders"]["dim"],
            signal_scale=cfg["encoders"].get("signal_scale", 1.0),
            distractor_std=cfg["encoders"].get("distractor_std", 0.0),
            seed=cfg["seed"],
        )

    # MC-MED
    if cfg["dataset_name"] == "mcmed":
        return DataModule_MCMED(
            batch_size = cfg["batch_size"],
            split_nr = cfg["split_nr"],
            max_waveform_windows = cfg["encoders"].get("max_waveform_windows", 0),
            radiology_model_params = cfg["encoders"].get("radiology", {}),
            require_all_modalities = cfg["encoders"].get("require_all_modalities", False),
            rad_for_main = cfg["encoders"].get("rad_for_main", "future"),
            require_both_rad_streams = cfg["encoders"].get("require_both_rad_streams", False),
        )

    # MIMIC (new trimodal variant)
    if cfg["dataset_name"] == "mimic":
        return DataModule_MIMIC(
            train_split_csv=cfg["encoders"]["train_split_csv"],
            val_split_csv=cfg["encoders"]["val_split_csv"],
            test_split_csv=cfg["encoders"]["test_split_csv"],
            mimic_root=cfg["encoders"]["mimic_root"],
            text_model_id=cfg["encoders"]["radiology"].get("text_model_id", "StanfordAIMI/RadBERT"),
            max_text_length=cfg["encoders"]["radiology"].get("max_text_length", 256),
            load_radiology_text=cfg["encoders"].get("load_radiology_text", True),
            require_all_modalities=cfg["encoders"].get("require_all_modalities", False),
            radiology_chunk_size=cfg["encoders"].get("radiology_chunk_size", 250000),
            batch_size=cfg["batch_size"],
            num_workers=cfg["encoders"].get("num_workers", 4),
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
