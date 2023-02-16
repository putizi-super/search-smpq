model=resnet50
mode_teacher=resnet50
flops=1
params=1
avg_bit_weights=2
avg_bit_fm=4

python 03-1-imagenet_finetuning_sqconv.py \
--arch $model \
--arch_teacher $mode_teacher \
--data_set imagenet \
--data_path /mnt/temp \
--job_dir ../../experiment/imagenet/$model/03-mix_precision_quntization/finetuning-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm/ \
--baseline True \
--baseline_model /home/lab611/workspace/xuke/pytorch/baseline/imagenet/resnet50-dali-apex-model_best.pth.tar \
--teacher_model  /home/lab611/workspace/xuke/pytorch/baseline/imagenet/resnet50-dali-apex-model_best.pth.tar \
--resume False \
--resume_model /home/lab611/workspace/xuke/pytorch/search-smpq/experiment/imagenet/mobilenet_v2/03-mix_precision_quntization/finetuning-w_bw-2-f_bw-2/checkpoint/model_checkpoint.pt \
--train_batch_size 64 \
--eval_batch_size 64 \
--optimizer SGD --lr 0.01 --weight_decay 1e-5 \
--label-smoothing 0.0 \
--num_epochs 60 \
--gpus 0 1 2

# 2>&1 | tee ../../experiment/temp/$model-search_fineturn-f-$flops-p-$params-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm-step.log
