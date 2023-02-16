model=mobilenet_v1
fp_flops=0.2
fp_params=0
target_fp_flops=0.50
target_fp_params=0.0
flops=1
params=1

python 02-0-imagenet_search_filter_pruning.py \
--arch $model \
--data_set imagenet \
--data_path /mnt/temp \
--job_dir ../../experiment/imagenet/$model/02-filter_pruning/search-flops-$target_fp_flops-params-$target_fp_params/ \
--baseline True \
--baseline_model /home/lab611/workspace/xuke/pytorch/baseline/imagenet/mobilenet_v1.pt \
--search_epochs 60 \
--train_batch_size 256 \
--eval_batch_size 256 \
-m_f_fp $fp_flops -m_p_fp $fp_params -t_f_fp $target_fp_flops -t_p_fp $target_fp_params \
-m_f $flops -m_p $params --no_fc False \
--gpus 0 
