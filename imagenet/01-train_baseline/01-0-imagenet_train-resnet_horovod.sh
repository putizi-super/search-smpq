model=resnet50

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 -H localhost:8  python 01-0-imagenet_train_baseline_horovod.py \
--data_set imagenet \
--data_path /mnt/data/imagenet \
--job_dir ../../experiment/imagenet/$model/baseline/ \
--arch $model \
--lr 0.1 \
--weight_decay 1e-4 \
--train_batch_size 64 \
--optimizer SGD \
--num_epochs 120 \
--warmup False --warmup_epochs 5 --label-smoothing 0.1