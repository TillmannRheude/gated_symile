#!/bin/bash

#SBATCH --job-name=Viz-Synthetic
#SBATCH -p gpu
#SBATCH --gres=shard:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-5

split_nr=${SLURM_ARRAY_TASK_ID}
seed=420

# p=0.0
# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="synthetic_xor" encoders="synthetic_xor" wandb.group="None" batch_size=128 devices=1 encoders.a_rule=xnor encoders.bc_corr_exclusive=True encoders.bc_corr_split=0.5 encoders.corr_modes=swap_signal encoders.distractor_std=3 encoders.embed_mode=u_v_uv encoders.n_bits=16 "encoders.p_corrs=[0, 0, 0]" "encoders.p_flips=[0, 0]" encoders.signal_scale=1 modelname.emb_dim=256 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=4.24612383392346 modelname.gate_temp=0.8331016757634284 modelname.gate_type=sigmoid modelname.logit_scale_init=-0.988811206378152 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.00240763284137706 optimizer.lr_gate_mul=11.357444406257558 optimizer.warmup_steps=200 optimizer.weight_decay=0.1

# p=0.2
# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="synthetic_xor" encoders="synthetic_xor" wandb.group="None" batch_size=128 devices=1 encoders.a_rule=xnor encoders.bc_corr_exclusive=True encoders.bc_corr_split=0.5 encoders.corr_modes=swap_signal encoders.distractor_std=3 encoders.embed_mode=u_v_uv encoders.n_bits=16 "encoders.p_corrs=[0, 0.2, 0.2]" "encoders.p_flips=[0, 0]" encoders.signal_scale=1 modelname.emb_dim=256 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=5.786598580876353 modelname.gate_temp=1.0655526594607054 modelname.gate_type=sigmoid modelname.logit_scale_init=-0.124048298092319 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.003963403626613062 optimizer.lr_gate_mul=2.344520810826637 optimizer.warmup_steps=200 optimizer.weight_decay=0.1

# p=0.4
# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="synthetic_xor" encoders="synthetic_xor" wandb.group="None" batch_size=128 devices=1 encoders.a_rule=xnor encoders.bc_corr_exclusive=True encoders.bc_corr_split=0.5 encoders.corr_modes=swap_signal encoders.distractor_std=3 encoders.embed_mode=u_v_uv encoders.n_bits=16 "encoders.p_corrs=[0, 0.4, 0.4]" "encoders.p_flips=[0, 0]" encoders.signal_scale=1 modelname.emb_dim=256 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=1.953827368865632 modelname.gate_temp=0.7858129643064544 modelname.gate_type=sigmoid modelname.logit_scale_init=-0.3057758139684088 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.008856477760567368 optimizer.lr_gate_mul=2.597386236229824 optimizer.warmup_steps=200 optimizer.weight_decay=0.1

# p=0.6
# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="synthetic_xor" encoders="synthetic_xor" wandb.group="None" batch_size=128 devices=1 encoders.a_rule=xnor encoders.bc_corr_exclusive=True encoders.bc_corr_split=0.5 encoders.corr_modes=swap_signal encoders.distractor_std=3 encoders.embed_mode=u_v_uv encoders.n_bits=16 "encoders.p_corrs=[0, 0.6, 0.6]" "encoders.p_flips=[0, 0]" encoders.signal_scale=1 modelname.emb_dim=256 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=4.962949481616641 modelname.gate_temp=0.7720851306589327 modelname.gate_type=sigmoid modelname.logit_scale_init=-0.31336749688055177 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.008630297330136285 optimizer.lr_gate_mul=1.812668667760312 optimizer.warmup_steps=100 optimizer.weight_decay=0.01

# p=0.8
# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="synthetic_xor" encoders="synthetic_xor" wandb.group="None" batch_size=128 devices=1 encoders.a_rule=xnor encoders.bc_corr_exclusive=True encoders.bc_corr_split=0.5 encoders.corr_modes=swap_signal encoders.distractor_std=3 encoders.embed_mode=u_v_uv encoders.n_bits=16 "encoders.p_corrs=[0, 0.8, 0.8]" "encoders.p_flips=[0, 0]" encoders.signal_scale=1 modelname.emb_dim=256 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=5.568838402708546 modelname.gate_temp=1.0312127513807017 modelname.gate_type=sigmoid modelname.logit_scale_init=-1.2280609205026742 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.007377935256943161 optimizer.lr_gate_mul=1.9992324469828215 optimizer.warmup_steps=200 optimizer.weight_decay=0.001

# p=1.0
# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="synthetic_xor" encoders="synthetic_xor" wandb.group="None" batch_size=128 devices=1 encoders.a_rule=xnor encoders.bc_corr_exclusive=True encoders.bc_corr_split=0.5 encoders.corr_modes=swap_signal encoders.distractor_std=3 encoders.embed_mode=u_v_uv encoders.n_bits=16 "encoders.p_corrs=[0, 1, 1]" "encoders.p_flips=[0, 0]" encoders.signal_scale=1 modelname.emb_dim=256 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=2.956870440185406 modelname.gate_temp=0.5057856129977136 modelname.gate_type=sigmoid modelname.logit_scale_init=-0.5524690048376106 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.008751964799911622 optimizer.lr_gate_mul=2.2698264264915595 optimizer.warmup_steps=200 optimizer.weight_decay=0


max_epochs=30
devices=1
patience=30
batch_size=128
negative_sampling="pair"
modelname_emb_dim=256
modelname_gate_d_k=256
modelname_gate_mode="attention"
modelname_gate_type="sigmoid"
modelname_neutral_type="random_trainable"
modelname_embedding_norm=True

# gate 
modelname_use_gate=True

# optim
# (set per p below)
modelname_optimizer_lr=0.0
modelname_optimizer_lr_gate_mul=1.0
modelname_optimizer_warmup_steps=0
modelname_optimizer_weight_decay=0.0

# encoders 
encoders_a_rule="xnor"
encoders_bc_corr_exclusive=True
encoders_bc_corr_split=0.5
encoders_corr_modes="swap_signal"
encoders_distractor_std=3
encoders_embed_mode="u_v_uv"
encoders_n_bits=16


# Set corruption based on array index: 0→[0,0,0], 1→[0,0.2,0.2], 2→[0,0.4,0.4], etc.
if [ "${split_nr}" -eq 0 ]; then
    p_val="0.0"
    encoders_p_corrs="[0,0.0,0.0]"
    modelname_logit_scale_init=-0.988811206378152
    modelname_gate_strength_init=4.24612383392346
    modelname_gate_temp=0.8331016757634284
    modelname_optimizer_lr=0.00240763284137706
    modelname_optimizer_lr_gate_mul=11.357444406257558
    modelname_optimizer_warmup_steps=200
    modelname_optimizer_weight_decay=0.1
elif [ "${split_nr}" -eq 1 ]; then
    p_val="0.2"
    encoders_p_corrs="[0,0.2,0.2]"
    modelname_logit_scale_init=-0.124048298092319
    modelname_gate_strength_init=5.786598580876353
    modelname_gate_temp=1.0655526594607054
    modelname_optimizer_lr=0.003963403626613062
    modelname_optimizer_lr_gate_mul=2.344520810826637
    modelname_optimizer_warmup_steps=200
    modelname_optimizer_weight_decay=0.1
elif [ "${split_nr}" -eq 2 ]; then
    p_val="0.4"
    encoders_p_corrs="[0,0.4,0.4]"
    modelname_logit_scale_init=-0.3057758139684088
    modelname_gate_strength_init=1.953827368865632
    modelname_gate_temp=0.7858129643064544
    modelname_optimizer_lr=0.008856477760567368
    modelname_optimizer_lr_gate_mul=2.597386236229824
    modelname_optimizer_warmup_steps=200
    modelname_optimizer_weight_decay=0.1
elif [ "${split_nr}" -eq 3 ]; then
    p_val="0.6"
    encoders_p_corrs="[0,0.6,0.6]"
    modelname_logit_scale_init=-0.31336749688055177
    modelname_gate_strength_init=4.962949481616641
    modelname_gate_temp=0.7720851306589327
    modelname_optimizer_lr=0.008630297330136285
    modelname_optimizer_lr_gate_mul=1.812668667760312
    modelname_optimizer_warmup_steps=100
    modelname_optimizer_weight_decay=0.01
elif [ "${split_nr}" -eq 4 ]; then
    p_val="0.8"
    encoders_p_corrs="[0,0.8,0.8]"
    modelname_logit_scale_init=-1.2280609205026742
    modelname_gate_strength_init=5.568838402708546
    modelname_gate_temp=1.0312127513807017
    modelname_optimizer_lr=0.007377935256943161
    modelname_optimizer_lr_gate_mul=1.9992324469828215
    modelname_optimizer_warmup_steps=200
    modelname_optimizer_weight_decay=0.001
elif [ "${split_nr}" -eq 5 ]; then
    p_val="1.0"
    encoders_p_corrs="[0,1.0,1.0]"
    modelname_logit_scale_init=-0.5524690048376106
    modelname_gate_strength_init=2.956870440185406
    modelname_gate_temp=0.5057856129977136
    modelname_optimizer_lr=0.008751964799911622
    modelname_optimizer_lr_gate_mul=2.2698264264915595
    modelname_optimizer_warmup_steps=200
    modelname_optimizer_weight_decay=0
else
    echo "Invalid split_nr=${split_nr} for encoders_p_corrs"
    exit 1
fi

encoders_p_flips="[0,0]"
encoders_signal_scale=1

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
        seed=${seed} \
        modelname="symile" \
        dataset_name="synthetic_xor" \
        encoders="synthetic_xor" \
        wandb.group="Viz-p${p_val}-Symile-${negative_sampling}-Synthetic-XOR" \
        devices=${devices} \
        datamodule.batch_size=${batch_size} \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        patience=${patience} \
        modelname.negative_sampling=${negative_sampling} \
        "encoders.p_corrs=${encoders_p_corrs}" \
        "encoders.p_flips=${encoders_p_flips}" \
        encoders.signal_scale=${encoders_signal_scale} \
        encoders.a_rule=${encoders_a_rule} \
        encoders.bc_corr_exclusive=${encoders_bc_corr_exclusive} \
        encoders.bc_corr_split=${encoders_bc_corr_split} \
        encoders.corr_modes=${encoders_corr_modes} \
        encoders.distractor_std=${encoders_distractor_std} \
        encoders.embed_mode=${encoders_embed_mode} \
        encoders.n_bits=${encoders_n_bits} \
        modelname.gate_d_k=${modelname_gate_d_k} \
        modelname.gate_mode=${modelname_gate_mode} \
        modelname.gate_strength_init=${modelname_gate_strength_init} \
        modelname.gate_temp=${modelname_gate_temp} \
        modelname.gate_type=${modelname_gate_type} \
        modelname.neutral_type=${modelname_neutral_type} \
        modelname.use_gate=${modelname_use_gate} \
        modelname.emb_dim=${modelname_emb_dim} \
        modelname.embedding_norm=${modelname_embedding_norm} \
        modelname.logit_scale_init=${modelname_logit_scale_init} \
        optimizer.lr=${modelname_optimizer_lr} \
        optimizer.lr_gate_mul=${modelname_optimizer_lr_gate_mul} \
        optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        optimizer.weight_decay=${modelname_optimizer_weight_decay}
