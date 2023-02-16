model=resnet56_fp
fp_rate_len=30
fp_flops=2.0
fp_params=1
flops=1
params=1
avg_bit_weights=32
avg_bit_fm=32
max_bw=8

CUDA_VISIBLE_DEVICES=2 python 02-0-search_filter_pruning_v1.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../../experiment/cifar10/$model/02-filter_pruning/v1/search \
--baseline True \
--baseline_model ../../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--search_epochs 60 \
-fp_rate_len $fp_rate_len -m_f_fp $fp_flops -m_p_fp $fp_params \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights True --avg_bw_weights $avg_bit_weights \
--fix_bw_fm True --avg_bw_fm $avg_bit_fm --max_bw $max_bw 
