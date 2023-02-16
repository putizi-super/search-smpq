model=resnet18
train_batch_size=128
eval_batch_size=100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
python -m torch.distributed.launch --nproc_per_node=8 01-1-imagenet_train_baseline_dali_apex.py \
--data_path /mnt/data/imagenet \
--job_dir ../../experiment/imagenet/resnet18/baseline-dali-apex/ \
--arch $model \
--train_batch_size $train_batch_size \
--eval_batch_size $eval_batch_size \
--optimizer adam \
--lr 1e-3 \
--weight_decay 1e-5 \
--num_epochs 120 \
--dali_cpu False \
--opt_level O2 \
--sync_bn \
--loss_scale dynamic \
--keep_batchnorm_fp32 True \
# --baseline ../../experiment/imagenet/resnet18/ir_bireal_O2sgdke/checkpoint/model_checkpoint.pt \
# --resume True \