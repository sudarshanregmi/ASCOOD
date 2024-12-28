# sh ./scripts/download/dowanload.sh

python ./scripts/download/download.py \
	--contents 'datasets' 'checkpoints' \
	--datasets 'ascood' \
	--checkpoints 'ascood' \
	--save_dir './data' './results' \
	--dataset_mode 'benchmark'
