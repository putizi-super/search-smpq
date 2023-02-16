model=resnet18

python quant-plot.py \
--data_path /home/lab611/data/imagenet \
--baseline True \
--arch $model \
--baseline_model /home/lab611/workspace/xuke/pytorch/search-smpq/experiment/imagenet/resnet18/finetuning-66.88%/checkpoint/model_best.pt \
--gpus 0 \
--train_batch_size 64