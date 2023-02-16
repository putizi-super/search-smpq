import os
import psutil
import os
import time
while True:
    if 32683 in psutil.pids():
        time.sleep(120)
    else:
        os.system('sh 03-1-smpq-correlation.sh')
        break
