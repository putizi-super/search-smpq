# Prune_Quant

Pytorch projects for CNN pruning and quantization

## TODO:

- [ ] 增加剪枝和量化的例子程序间一个example目录，以Resnet50为例给一个剪枝量化的例子
- [ ] 添加每个程序的详细使用方法说明、参数意义（如何设置）
- [ ] 给出服务器上保存的已量化和剪枝模型路径

***

## 程序功能说明

基于该框架训练一个剪枝/量化模型可分为三步：

1. 训练一个对应模型的全精度baseline（若有baseline可省略这一步）
2. 搜索剪枝率/量化比特数（量化方法采用LSQ）
3. 基于搜索的剪枝率/量化比特数进行finetuning

另外该框架还支持基于Ristretto方法的8bit量化，见04号程序

***

```sh
├── cifar-10 # cifar10剪枝量化，此模块为cifar-10部分主函数
├── data # 数据集处理
├── experiment # 实验结果
├── imagenet # imagenet剪枝量化，此模块为imagenet部分主函数
│   ├── 00-test
│   ├── 01-train_baseline
│   ├── 02-filter_pruning
│   ├── 03-mix_precision_quantization
│   └── 04-ristretto_quantization
├── loss # Loss函数
├── models # 模型构建
├── modules # 量化模型
│   ├── ristretto # ristretto量化
│   └── sq_module # LSQ量化
├── search # 搜索算法（这部分与search-prj内容有重复）
│   ├── ABC # 人工蜂群算法
│   ├── ACO # 蚁群算法
│   ├── GA # 遗传算法
│   └── PSO # 粒子群算法
├── tool # merge bn & 画图工具
│   ├── pruning # 剪枝
│   ├── thop 
│   ├── histogram.py # 直方图统计
│   ├── mergebn.py # merge BN工具
│   ├── meter.py # 参数量，计算量统计
│   └── quant-plot.py # 权重激活分布直方图
├── tricks # 训练技巧
└── utils # 通用操作
│   ├── balance.py # 多卡训练平衡数据
│   ├── common.py # 权重获取与加载
│   ├── options.py # 传参
│   └── summary.py 
```

## 1. Training baseline

```
01-train_baseline
```

以 `01-0-imagenet_train-mobilenet.sh` 为例

```
model=mobilenetv2_fp # 调用模型，此脚本适用于mobienet系列模型，其他采用01-0-imagenet_train-resnet.sh

python 01-0-imagenet_train_baseline.py \
--data_set imagenet \ # 数据集类型
--data_path /mnt/data/imagenet \ # 数据集路径
--job_dir ../../experiment/imagenet/$model/baseline/ \ # baseline 保存路径
--arch $model \
--lr 0.05 \ # 学习率
--weight_decay 4e-5 \ 
--train_batch_size 256 \
--optimizer SGD \ # 优化器
--num_epochs 150 \ # 训练轮数
--warmup False --warmup_epochs 5 --label-smoothing 0.1 \ # 训练技巧
--gpus 0 1 2 3
```

## 2. Search for pruning rate and finetuning

```
02-filter_pruning
```

### search for Pruning rate

```
02-0-imagenet_search_filter_pruning.sh
```

You can set the pruning rate by changing `flops` and `params`, `flops` means the pruning rate of the amount of calculation and `params` represents parameters, if you set them to `1`, just means no pruning

An example of search results, which will be printed on the screen, it shows the pruning rate from the first layer to the last layer:

```
best_candidates_prune=[0.2509892381838768, 0.1949955777182775, 0.15481105457277416, 0.181482879280324, 0.15236411504078137, 0.12624166044322543, 0.19945650833123205, 0.14867201615531073, 0.09744269325603885, 0.16507781611943093, 0.22461711527148995, 0.19174842125328845, 0.13530936319651926, 0.2769501456887007, 0.1281951215877744, 0.09779858333620649, 0.13675934743598922, 0.18702198771408263, 0.2592740954241761, 0.16431281448915383, 0.15246512653703692, 0.2972427183432819]

```

### finetuning

```
02-1-imagenet_finetuning_filter_pruning.sh 
```

When finetuning, it is needed to load baseline, so set `baseline` to `True` and set `baseline_model` to the path of baseline, and you need to set `pruning_rate` to the same as `best_candidates_prune` printed on the screen after running `02-0-imagenet_search_filter_pruning.sh`, The last pruned baseline will be saved automatically for loading.

### example for MobieNetV1

```
sh 02-0-imagenet_search_filter_pruning.sh
copy best_candidates_prune(printed on the screen after running 02-0-imagenet_search_filter_pruning.sh) to 02-1-imagenet_finetuning_filter_pruning.py
sh 02-1-imagenet_finetuning_filter_pruning.sh

```

## 3. search for quantization bits and finetuning

```
03-mix_precision_quntization
```

### search for quantization bits

You can set quantization bits of weights by `avg_bit_weights`, and `avg_bit_fm` is for activations, But you need to note that they are average bits, so the number of bits per layer may be different, if you want the number of bits in each layer to be same, you can set `fix_bw_weights` and `fix_bw_fm` to `True`.

An example of search results, which will be printed on the screen, it shows the quantization bits of weights and activations from the first layer to the last layer, 
The pruning rate search is still retained here, if you do not need it, you can set it to `1`:

```
w_bit=[8.0, 7.0, 6.0, 7.0, 8.0, 5.0, 3.0, 4.0, 3.0, 6.0, 3.0, 4.0, 3.0, 4.0, 6.0, 3.0, 2.0, 3.0, 3.0, 1.0, 8.0]
a_bit=[6.0, 3.0, 3.0, 2.0, 2.0, 2.0, 3.0, 2.0, 5.0, 1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 4.0, 2.0, 2.0, 5.0, 7.0, 8.0]
pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### finetuning

It is same as finetuning in pruning, the difference is that setting `w_bit` and `a_bit` that you have searched.The last quantified baseline will be saved automatically for loading, The relevant parameters of LSQ quantization method are stored in the baseline.

## 4. ristretto quantization

`04-mergebn_ristretto.py` is for quatization by ritretto after merge BN, and `04-ristretto.py` is just for quatization by ritretto, `fl` will be printed on the screen after running it.

(`ResNet18-caffe`, `ResNet50-caffe` are suitable for quatization by ritretto after merge BN, and weights can be quantified by unsaturation, but `MobileNetV1` and `MobileNetV2` can't be quantified by ristretto after merge BN)

## 5. merge BN

```
.\tool\mergebn.py

if __name__ == "__main__":
    params = get_params_mergebn(model) # 得到原始模型merge BN后参数
    model = fillbias(model) # 为原始模型加上偏置
    model = convert_to_mergebn(model) # 去掉原始模型中BN层
    model = utils.load_params_model(model,params) # 把merge BN后的参数加载进修改后的模型
```

***

## 实验结果

|     model      | baseline | ristretto/8bit |  pruning  | pruning+8bit |
| :------------: | :------: | :------------: | :-------: | :----------: |
| ResNet18-caffe |  71.05   |     70.44      | 68.90(X6) |    68.34     |
| ResNet50-caffe |  76.63   |     76.07      | 75.32(X6) |    75.04     |
|  MobileNetV1   |  71.37   |     70.11      | 70.02(X2) |    69.25     |
|  MobileNetV2   |  71.78   |     70.93      | 70.47(X2) |    70.10     |
|   DarkNet19    |  72.58   |                | 72.02(X6) |              |
|   DarkNet53    |  75.10   |                | 75.10(X6) |              |

*MobileNetV1,V2 do not merge BN*
*ResNet50-caffe和原版区别：resnet-50-caffe 是根据caffe默认resnet-50网络配置训练得来(通过比对pytorch打印出来的resnet50的model与caffe prototxt的网络结构发现：网络的第12、13、25、26、44、45层的stride，pytorch版是前面层是1，下一层为2；而caffe是前面层是2，下一层为1)，这样做可以减少硬件feature buffer大小*

## 保存模型

```
在50服务器上，路径
/home/ic611/Work2/MyModelZoo/
```"# search-smpq" 
