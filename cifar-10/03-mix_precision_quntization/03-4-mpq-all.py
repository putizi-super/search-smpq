#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xlwt
import re
import os

# http://www.txt2re.com 自动生成代码
def decode_log(txt):
    re1='.*?'	# Non-greedy match on filler
    re2='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 1
    re3='.*?'	# Non-greedy match on filler
    re4='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 2
    re5='.*?'	# Non-greedy match on filler
    re6='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 3
    re7='.*?'	# Non-greedy match on filler
    re8='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 4
    re9='.*?'	# Non-greedy match on filler
    re10='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 5
    re11='.*?'	# Non-greedy match on filler
    re12='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 6
    re13='.*?'	# Non-greedy match on filler
    re14='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 7
    re15='.*?'	# Non-greedy match on filler
    re16='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 8
    re17='.*?'	# Non-greedy match on filler
    re18='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 9
    re19='.*?'	# Non-greedy match on filler
    re20='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'	# Float 10

    rg1 = re.compile(re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+\
        re11+re12+re13+re14+re15+re16+re17+re18+re19+re20,re.IGNORECASE|re.DOTALL)
    #rg2 = re.compile(re1+re2+re3+re4+re5+re6,re.IGNORECASE|re.DOTALL)
    m = rg1.search(txt)
    #m2 = rg2.search(txt)
    data = []
    if m:
        Flops=float(m.group(1))
        Pruning_Flops=float(m.group(2))
        Params=float(m.group(3))
        Pruning_Params=float(m.group(4))
        avg_Bw=float(m.group(5))
        Pruning_Bw=float(m.group(6))
        avg_Ac=float(m.group(7))
        Pruning_Fw=float(m.group(8))
        search_acc=float(m.group(9))
        fineturn_acc=float(m.group(10))
        
        data.append(Flops)
        data.append(Pruning_Flops)
        data.append(Params)
        data.append(Pruning_Params)
        data.append(avg_Bw)
        data.append(Pruning_Bw)
        data.append(avg_Ac)
        data.append(Pruning_Fw)
        data.append(search_acc)
        data.append(fineturn_acc)
    return data

def get_params(fname):
    res = [b'Sparse FLOPs']
    with open(fname,'rb') as fname:
        lines=fname.readlines()
        params = []
        for line in lines:
            pt=line.find(res[0])
            if(pt!=-1):
                line = str(line)
                # print(line[1:])
                data = decode_log(line[1:])
                params.append(data)
    return params

def main():
    flops = 1
    params=1
    f_bws = [2]
    w_bws = [2]
    gpu_id = 0
    model = 'resnet20'

    workbook = xlwt.Workbook(encoding = 'utf-8')

    for f_bw in f_bws:
        for w_bw in w_bws:
            os.system('sh 03-3-mpq-ex.sh'+ ' ' +str(model)+ ' ' + str(w_bw)+ ' ' + str(f_bw) + ' ' + str(gpu_id))

            result_sheet = workbook.add_sheet('w_bw-'+str(w_bw)+'-f_bw-'+str(f_bw))
            style = xlwt.XFStyle()
            alignment = xlwt.Alignment()
            alignment.horz = 0x02
            alignment.vert = 0x01
            style.alignment = alignment
            LAB_REPORT=['FLOPS','Pruning-Flops','Params','Pruning-Params','Avg-BW-Weights','Pruning-Bw','Avg-BW-Activation','Pruning-Fw','search-Acc','Fineturn-Acc']
            for i in range(len(LAB_REPORT)):
                result_sheet.write(0,i, LAB_REPORT[i], style)
            fname = '../../experiment/cifar10/'+str(model)+'/03-mix_precision_quntization/ex/search_fineturn-f-'+str(flops)+'-p-'+str(params)+'-w_bw-'+str(w_bw)+'-f_bw-'+str(f_bw)+'/'+str(model)+'-search_fineturn-f-'+str(flops)+'-p-'+str(params)+'-w_bw-'+str(w_bw)+'-f_bw-'+str(f_bw)+'.log'
            net_params = get_params(fname)
            for i, net_param, in enumerate(net_params):
                for j in range(len(net_param)):
                    result_sheet.write(1+i,j, net_param[j],style)

            workbook.save('../../experiment/cifar10/'+str(model)+'/03-mix_precision_quntization/ex/search_fineturn-f-'+str(flops)+'-p-'+str(params)+'-w_bw-'+str(w_bw)+'-f_bw-'+str(f_bw)+'/result.xls')

if __name__ == "__main__":
   main()