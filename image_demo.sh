model=vision_transformer
wbit=4
abit=8



python image_demo.py    \
--arch $model \
--img "demo/demo.JPEG"    \
--config "vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py"    \
--checkpoint "baseline\imagenet\vit-base.pth"   \
--data_path ./dataset/ImageNet \
--data_set  imagenet \
--data_path ./dataset/ImageNet \
--job_dir ./out/imagenet/$model/brecq/ \

# --device "cpu"  \
--w_bit $wbit \
--w_bit $wbit \
--a_bit $abit \
