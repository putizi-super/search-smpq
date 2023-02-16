model=resnet20
flops=1
params=1
avg_bit_weights=2
avg_bit_fm=2
max_bw=6
min_bw=1

CUDA_VISIBLE_DEVICES=0 python 03-2-mpq-correlation.py \
--arch $model \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data/cifar-10 \
--job_dir ../../experiment/cifar10/$model/03-mix_precision_quntization/correlation/ \
--baseline True \
--baseline_model ../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
--search_epochs 20 \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights False --avg_bw_weights $avg_bit_weights \
--fix_bw_fm False --avg_bw_fm $avg_bit_fm --max_bw $max_bw --min_bw $min_bw \
--optimizer SGD --lr 0.01 --weight_decay 1e-5 \
--label-smoothing 0.0 \
--num_epochs 150 \
2>&1 | tee ../../experiment/$model-search_fineturn-f-$flops-p-$params-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm-correlation-Adeptive-All.log