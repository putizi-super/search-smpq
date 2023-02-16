# --baseline True \--channel_wise \
# --baseline_model /home/ic611/workspace/moruchan/baseline/imagenet/mobilenetv2-dali-apex-model_best.pth.tar \
# /home/ic611/workspace/moruchan/baseline/imagenet/darknet19.pt
# /home/ic611/workspace/moruchan/baseline/imagenet/reactnet_mobilenet.tar

# model=darknet19
model=resnet18
wbit=4
abit=8

python 04-brecq_ristretto.py \
--arch $model \
--data_set  imagenet \
--data_path /home/putizi/dataset/imagenet \
--job_dir ../experiment/imagenet/$model/brecq/ \
--baseline True \
--baseline_model /home/ic611/workspace/moruchan/baseline/imagenet/darknet19.pt \
--eval_batch_size 128 \
--w_bit $wbit \
--a_bit $abit \
--act_quant --test_before_calibration \
# --gpus 2
--gpus 0