model=mobilenet_v3_small

CUDA_VISIBLE_DEVICES=2 python 00-test.py \
--arch $model \
--data_set imagenet \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../experiment/imagenet/$model/00-test/ \
--baseline True \
--baseline_model ../../experiment/imagenet/$model/baseline/checkpoint/model_best.pt  \
--lr 0.01 \
--label-smoothing 0.0 \
--num_epochs 150 \

# --baseline False \
# --baseline_model ../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
