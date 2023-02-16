model=resnet20_fp
fp_rate_len=12
fp_flops=2
fp_params=1

CUDA_VISIBLE_DEVICES=0 python 02-2-filter_pruning_correlation_v1.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../../experiment/cifar10/$model/02-filter_pruning/v1/correlation \
--baseline True \
--baseline_model ../../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
-fp_rate_len $fp_rate_len -m_f_fp $fp_flops -m_p_fp $fp_params \
--lr 0.001 --weight_decay 0 \
--label-smoothing 0.0 \
--num_epochs 150 \
