model=resnet20_B

CUDA_VISIBLE_DEVICES=2 python 01-train_baseline.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../experiment/cifar10/$model/baseline/ \
--optimizer SGD --lr 0.1 \
--label-smoothing 0.0 \
--autoaug True --cutout True --cutout_length 16 \
--num_epochs 150 \
