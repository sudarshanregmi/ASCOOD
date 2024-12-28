#!/bin/bash

python scripts/eval_ood.py \
   --id-data cars \
   --wrapper-net ASCOODNet \
   --root ./results/cars_ascood_net_ascood_e30_lr0.01_w1.0_p0.1_otype_gradient_nmg_0.1_0.1_default \
   --postprocessor relation --save-score --save-csv
