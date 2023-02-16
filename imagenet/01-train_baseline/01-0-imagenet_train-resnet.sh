model=resnet18

python 01-0-imagenet_train_baseline.py \
--data_set imagenet \
--data_path /mnt/data/imagenet \
--job_dir ../../experiment/imagenet/$model/baseline/ \
--arch $model \
--lr 0.1 \
--weight_decay 1e-4 \
--train_batch_size 512 \
--optimizer SGD \
--num_epochs 90 \
--dali False --workers 32 \
--warmup True --warmup_epochs 5 \
--autoaug True --cutout True --cutout_length 16 --label-smoothing 0.1 \
--gpus 4 5 6 7