model=resnet56_B

CUDA_VISIBLE_DEVICES=3 python 02-1-finetuning_filter_pruning_v2.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data/cifar-10 \
--job_dir ../../../experiment/cifar10/$model/02-filter_pruning/v2/finetuning/ \
--baseline True \
--baseline_model ../../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--optimizer SGD --lr 0.01 --weight_decay 5e-4 \
--autoaug True --cutout False --cutout_length 16 --label-smoothing 0.0 \
--num_epochs 200 

# --optimizer SGD --lr 0.01 --weight_decay 5e-4 
