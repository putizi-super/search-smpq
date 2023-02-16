model=resnet50
flops=1
params=1
avg_bit_weights=2
avg_bit_fm=2
max_bw=5
min_bw=1

python 03-2-imagenet_correlation_sqconv.py \
--arch $model \
--data_set imagenet \
--data_path /mnt/data/imagenet \
--job_dir ../../experiment/imagenet/$model/03-mix_precision_quntization/correlation-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm/ \
--baseline True \
--baseline_model /home/ubuntu/workspace/xuke/baseline/imagenet/resnet50-dali-apex-model_best.pth.tar \
--search_epochs 20 \
--num_epochs 10 --workers 32 \
--train_batch_size 128 \
--eval_batch_size 128 \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights False --avg_bw_weights $avg_bit_weights \
--fix_bw_fm False --avg_bw_fm $avg_bit_fm --max_bw $max_bw --min_bw $min_bw \
--optimizer Adam --lr 1e-4 --weight_decay 1e-5 \
--gpus 0 1 2 3 4 5 6 7 \
2>&1 | tee ../../experiment/temp/$model-correlation-f-$flops-p-$params-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm.log