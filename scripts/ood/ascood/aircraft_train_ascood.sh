#!/bin/bash

python main.py \
    --config configs/datasets/aircraft/aircraft.yml \
    configs/networks/ascood_net.yml \
    configs/pipelines/train/train_ascood.yml \
    configs/preprocessors/ascood_preprocessor.yml \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.01 \
    --optimizer.num_epochs 30 \
    --trainer.trainer_args.alpha_min 0.1 \
    --trainer.trainer_args.alpha_max 0.1 "$@"
