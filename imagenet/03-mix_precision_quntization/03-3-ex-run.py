# import os
# import psutil
# import os
# import time
# while True:
#     if 3272 in psutil.pids():
#         time.sleep(120)
#     else:
#         lr_list = [5e-4]
#         wd_list = [0, 1e-5]
#         for lr in lr_list:
#             for wd in wd_list:
#                 os.system('sh 05-imagenet_finetuning_sqconv.sh'+ ' ' +str(lr)+ ' ' + str(wd))

import os
import psutil
import os
import time
while True:
    if 14222 in psutil.pids():
        time.sleep(120)
    else:
        os.system('sh 03-imagenet_finetuning_sqconv.sh')
        break
