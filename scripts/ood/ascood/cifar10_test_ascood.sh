#!/bin/bash

python scripts/eval_ood.py \
   --id-data cifar10 \
   --wrapper-net ASCOODNet \
   --root ./results/cifar10_ascood_net_ascood_e100_lr0.1_w1.0_p0.2_otype_shuffle_nmg_1.0_1.0_random \
   --postprocessor scale --save-score --save-csv
