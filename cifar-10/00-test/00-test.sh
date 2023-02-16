model=resnet20

CUDA_VISIBLE_DEVICES=0 python 00-test.py \
--arch $model \
--data_set cifar10 \
--data_path /mnt/data/cifar-10 \
--job_dir ../../experiment/cifar10/$model/00-test/ \
--baseline True \
--baseline_model ../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--lr 0.01 \
--label-smoothing 0.0 \
--num_epochs 150 \

# --baseline False \
# --baseline_model ../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
