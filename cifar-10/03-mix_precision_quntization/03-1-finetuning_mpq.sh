model=resnet20
model_teacher=resnet56_fp
flops=1
params=1
avg_bit_weights=2
avg_bit_fm=2

CUDA_VISIBLE_DEVICES=3 python 03-1-finetuning_mpq.py \
--arch $model \
--arch_teacher $model_teacher \
--data_set cifar10 \
--data_path /home/lab611/workspace/xuke/pytorch/data/cifar-10 \
--job_dir ../../experiment/cifar10/$model/03-mix_precision_quntization/finetuning/ \
--baseline True \
--baseline_model ../../experiment/cifar10/$model/baseline/checkpoint/model_best.pt \
--teacher_model ../../experiment/cifar10/$model_teacher/baseline/checkpoint/model_best.pt \
--optimizer SGD --lr 0.01 --weight_decay 1e-5 \
--label-smoothing 0.0 \
--num_epochs 150 \
2>&1 | tee ../../experiment/$model-search_fineturn-f-$flops-p-$params-w_bw-$avg_bit_weights-f_bw-$avg_bit_fm-test.log
