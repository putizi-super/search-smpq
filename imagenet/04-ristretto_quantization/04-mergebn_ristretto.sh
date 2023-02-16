model=resnet18_caffe
avg_bit_weights=8
avg_bit_fm=8

python 04-mergebn_ristretto.py \
--arch $model \
--data_set imagenet \
--data_path /home/ic611/Work2/ImageNet \
--job_dir ../experiment/imagenet/$model/ristretto_test/ \
--baseline True \
--baseline_model /home/ic611/workspace/moruchan/baseline/imagenet/resnet18-caffe-dali-apex-model_best.pth.tar \
--train_batch_size 128 \
--eval_batch_size 128 \
--no_fc False \
--fix_bw_weights True --avg_bw_weights $avg_bit_weights \
--fix_bw_fm True --avg_bw_fm $avg_bit_fm \
--optimizer Adam --lr 1e-3 --weight_decay 1e-5 \
--label-smoothing 0.0 \
--num_epochs 40 \
--gpus 0 1 2 3