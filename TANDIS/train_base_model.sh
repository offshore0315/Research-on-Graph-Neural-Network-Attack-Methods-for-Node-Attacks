#!/bin/bash

dataset=cora
model_name=Deep
layer=sage
down_task=node_classification


model_save_name=model-${model_name}-${layer} #${layer} #-lcc #-featpre #-${layer} #-${hidden_dim}

hidden_dim=64
directed=0
dropout_rate=0.5
n_epochs=100
lr=0.01
batch_size=20
patience=60
save_embs=0

cd attack_models/

output_root=${down_task}/${dataset}

if [ ! -d $output_root ];
then
    mkdir -p $output_root
fi

output_root=emb_models/${dataset}

if [ ! -d $output_root ];
then
    mkdir -p $output_root
fi

# if [ $model_name == Deep ]; then
#     runFile=train_model.py
# elif [ $model_name == PGNN ]; then
#     runFile=train_PGNN.py
# elif [ $model_name == GNNGuard ]; then
#     runFile=train_gnnGuard.py
# fi

python train_model.py \
    -dataset $dataset \
    -model_name $model_name \
    -layer $layer \
    -down_task $down_task \
    -model_save_name $model_save_name \
    -dropout_rate $dropout_rate \
    -n_epochs $n_epochs \
    -hidden_layers ${hidden_dim} ${hidden_dim} \
    -directed $directed \
    -lr $lr \
    -patience $patience \
    -save_embs $save_embs \
    -batch_size $batch_size \
    -device cuda \
    -lcc \
    $@

# rm temp*.pt
