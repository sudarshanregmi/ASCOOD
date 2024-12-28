#!/bin/bash

python main.py \
    --config configs/datasets/waterbirds/waterbirds.yml \
    configs/networks/ascood_net.yml \
    configs/pipelines/train/train_ascood.yml \
    configs/preprocessors/ascood_preprocessor.yml \
    --network.backbone.name resnet18_224x224 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet18-f37072fd.pth \
    --optimizer.lr 0.01 \
    --trainer.trainer_args.w 0.1 \
    --trainer.trainer_args.alpha_max 300.0 \
    --trainer.trainer_args.alpha_min 30.0 \
    --optimizer.num_epochs 30 "$@"
