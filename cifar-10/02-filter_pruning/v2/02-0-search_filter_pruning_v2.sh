model=resnet56_B
fp_flops=0.1
fp_params=0
target_fp_flops=0.52
target_fp_params=0.0
flops=1
params=1
avg_bit_weights=32
avg_bit_fm=32
max_bw=8

CUDA_VISIBLE_DEVICES=3 python 02-0-search_filter_pruning_v2.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data/cifar-10 \
--job_dir ../../../experiment/cifar10/$model/02-filter_pruning/v2/search/ \
--baseline True \
--baseline_model ../../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--search_epochs 60 \
-m_f_fp $fp_flops -m_p_fp $fp_params -t_f_fp $target_fp_flops -t_p_fp $target_fp_params \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights True --avg_bw_weights $avg_bit_weights \
--fix_bw_fm True --avg_bw_fm $avg_bit_fm --max_bw $max_bw 
