model=resnet20_B
fp_flops=0.2
fp_params=0
target_fp_flops=0.52
target_fp_params=0.0
flops=1
params=1
avg_bit_weights=32
avg_bit_fm=32
max_bw=8

CUDA_VISIBLE_DEVICES=3 python 02-2-filter_pruning_correlation_v2.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../../experiment/cifar10/$model/02-filter_pruning/v2/correlation \
--baseline True \
--baseline_model ../../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
-m_f_fp $fp_flops -m_p_fp $fp_params -t_f_fp $target_fp_flops -t_p_fp $target_fp_params  \
--optimizer SGD --lr 0.01 --weight_decay 1e-5  \
--label-smoothing 0.0 \
--num_epochs 150 \
