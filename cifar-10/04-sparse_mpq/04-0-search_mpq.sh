model=resnet20
flops=5
params=5
avg_bit_weights=3
avg_bit_fm=8
max_bw=6
min_bw=2

CUDA_VISIBLE_DEVICES=0 python 04-0-search_mpq.py \
--arch $model \
--data_set cifar10 \
--train_batch_size 128 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../experiment/cifar10/$model/04-sparse_mpq/search/ \
--baseline True \
--baseline_model ../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--search_epochs 60 \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights False --avg_bw_weights $avg_bit_weights \
--fix_bw_fm True --avg_bw_fm $avg_bit_fm --max_bw $max_bw  --min_bw $min_bw 
