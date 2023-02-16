model=resnet18
flops=1
params=1
avg_bit_weights=2
avg_bit_fm=2
max_bw=6
min_bw=1

python 03-0-imagenet_search_sqconv.py \
--arch $model \
--data_set imagenet \
--data_path /home/lab611/data/imagenet \
--job_dir ../../experiment/imagenet/$model/03-mix_precision_quntization/search-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm/ \
--baseline True \
--baseline_model /home/lab611/workspace/xuke/pytorch/baseline/imagenet/resnet18-dali-apex-model_best.pth.tar \
--search_epochs 60 \
--train_batch_size 128 \
--eval_batch_size 128 \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights False --avg_bw_weights $avg_bit_weights \
--fix_bw_fm False --avg_bw_fm $avg_bit_fm --max_bw $max_bw --min_bw $min_bw \
--gpus 0 1 3
# 2>&1 | tee ../../experiment/$model-search-f-$flops-p-$params-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm.log
