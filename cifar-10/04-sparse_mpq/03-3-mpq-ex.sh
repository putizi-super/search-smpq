flops=1
params=1
max_bw=6
min_bw=1

CUDA_VISIBLE_DEVICES=$4 python 03-3-mpq-ex.py --arch $1 --avg_bw_weights $2 --avg_bw_fm $3 \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data \
--job_dir ../../experiment/cifar10/$1/03-mix_precision_quntization/ex/search_fineturn-f-$flops-p-$params-w_bw-$2-f_bw-$3/ \
--baseline True \
--baseline_model ../../experiment/cifar10/$1/baseline/checkpoint/model_best.pt \
--train_batch_size 128 \
--eval_batch_size 100 \
--optimizer SGD --lr 0.01 --weight_decay 1e-5 \
--label-smoothing 0.0 \
--num_epochs 150 \
--search_epochs 60 \
-m_f $flops -m_p $params --no_fc False \
--fix_bw_weights False \
--fix_bw_fm False --max_bw $max_bw  --min_bw $min_bw \
2>&1 | tee ../../experiment/temp/$1-search_fineturn-f-$flops-p-$params-w_bw-$2-f_bw-$3.log

mv ../../experiment/temp/$1-search_fineturn-f-$flops-p-$params-w_bw-$2-f_bw-$3.log ../../experiment/cifar10/$1/03-mix_precision_quntization/ex/search_fineturn-f-$flops-p-$params-w_bw-$2-f_bw-$3/