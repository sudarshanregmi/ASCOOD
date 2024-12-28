#!/bin/bash

python scripts/eval_ood.py \
   --id-data waterbirds \
   --wrapper-net ASCOODNet \
   --root ./results/waterbirds_ascood_net_ascood_e30_lr0.01_w0.1_p0.1_otype_gradient_nmg_30.0_300.0_default \
   --postprocessor odin --save-score --save-csv
