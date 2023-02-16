model=mobilenetv2_fp

python 01-0-imagenet_train_baseline.py \
--data_set imagenet \
--data_path /mnt/data/imagenet \
--job_dir ../../experiment/imagenet/$model/baseline/ \
--arch $model \
--lr 0.05 \
--weight_decay 4e-5 \
--train_batch_size 256 \
--optimizer SGD \
--num_epochs 150 \
--warmup False --warmup_epochs 5 --label-smoothing 0.1 \
--gpus 0 1 2 3