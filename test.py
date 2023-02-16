import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

baseline_model = '/home/putizi/baseline/imagenet/darknet19.pth'
ckpt = torch.load(baseline_model)
# b = str(ckpt)
# with open("test.txt","w") as f:
#     f.write(b)
baseline_key = list(ckpt.keys())
print(baseline_key)