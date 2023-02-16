model=resnet56_fp

CUDA_VISIBLE_DEVICES=2 python 02-1-finetuning_filter_pruning_v1.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../../experiment/cifar10/$model/02-filter_pruning/v1/finetuning \
--baseline True \
--baseline_model ../../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--optimizer Adam --lr 1e-3 --weight_decay 5e-4 \
--label-smoothing 0.0 \
--num_epochs 120 
