model=mobilenet_v1
target_fp_flops=0.50
target_fp_params=0.0

python 02-1-imagenet_finetuning_filter_pruning.py \
--arch $model \
--data_set imagenet \
--data_path /mnt/temp \
--job_dir ../../experiment/imagenet/$model/02-filter_pruning/finetuning-flops-$target_fp_flops-params-$target_fp_params/ \
--baseline True \
--baseline_model /home/lab611/workspace/xuke/pytorch/baseline/imagenet/mobilenet_v1.pt \
--resume False \
--resume_model ../../experiment/imagenet/$model/finetuning_fp/checkpoint/model_checkpoint.pt \
--train_batch_size 512 \
--eval_batch_size 512 \
--optimizer SGD --lr 0.01 --weight_decay 1e-5 \
--label-smoothing 0.0 \
--num_epochs 90 \
--gpus 0 1 2 
