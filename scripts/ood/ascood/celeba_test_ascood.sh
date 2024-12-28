#!/bin/bash

python scripts/eval_ood.py \
   --id-data celeba \
   --wrapper-net ASCOODNet \
   --root ./results/celeba_ascood_net_ascood_e30_lr0.01_w1.0_p1.0_otype_gaussian_nmg_0.1_0.1_default \
   --postprocessor odin --save-score --save-csv

python scripts/eval_ood.py \
   --id-data celeba \
   --wrapper-net ASCOODNet \
   --root ./results/celeba_ascood_net_ascood_e30_lr0.01_w1.0_p0.05_otype_gradient_nmg_30.0_50.0_default \
   --postprocessor odin --save-score --save-csv
