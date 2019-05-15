import numpy
import os, sys, time
import math,copy
from  multiprocessing import Process
import re
import csv

processes_num = 16

def os_csv_path(rootdir, file_list):#tell a dir path ,get all the files' name and put in a list
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
         path = os.path.join(rootdir,list[i])
         if os.path.isfile(path) and (".csv" in path):
             file_list.append(path)
         if os.path.isdir(path):
             os_csv_path(path, file_list)


def open_csv(path):#read the csv
    first_line = numpy.loadtxt(path, str, delimiter= ',')
    return first_line

def csv_read(loadcsv_path):
    loadcsv = numpy.loadtxt(loadcsv_path, str, delimiter= ',')
    return loadcsv

def filelist_test(num): #interface of the function which will create the fileprocess by user define
    file_list = []
    os_csv_path("/ssd/diskcsv", file_list)
    all_li = create_fileprocess(file_list, num)
    for i in all_li:
        for j in i:
            print(j)


def create_fileprocess(file_list, proce_num): #create the fileprocess by user define
    fileList_model = list()
    fileList_all = list()
    for i in range(int(proce_num)):
        fileList_copy = copy.deepcopy(fileList_model)
        fileList_all.append(fileList_copy)
    for i in range(len(file_list)):
        l = i%int(proce_num)
        for j in range(int(proce_num)):
            if j == l:
                fileList_all[l].append(file_list[i])
    return fileList_all


def write_csv(file_list, input_csvfile):
    with open(input_csvfile, 'w', newline='') as f:
        writer = csv.writer(f)
        select_useful_item(file_list, writer)

def try_int(line, index):
    if (index + 1):
        try:
            ret = int(line[index])
            return ret
        except:
            return 0
    else:
        return 0

def select_useful_item(file_list, writer):
    for csv in file_list:
        print(csv)
        diskinfo = csv_read(csv)
        title = list(diskinfo[0])
        try:
            model_index = title.index("model")
        except:
            model_index = -1
        try:
            failure_index = title.index("failure")
        except:
            failure_index = -1

        try:
            smart2_index = title.index("smart_2_raw")
        except:
            smart2_index = -1
        try:
            smart3_index = title.index("smart_3_raw")
        except:
            smart3_index = -1
        try:
            smart4_index = title.index("smart_4_raw")
        except:
            smart4_index = -1
        try:
            smart195_index = title.index("smart_195_raw")
        except:
            smart195_index = -1
        try:
            smart193_index = title.index("smart_193_raw")
        except:
            smart193_index = -1
        try:
            smart194_index = title.index("smart_194_raw")
        except:
            smart194_index = -1

        try:
            smart5_index = title.index("smart_5_raw")
        except:
            smart5_index = -1
        try:
            smart9_index = title.index("smart_9_raw")
        except:
            smart9_index = -1
        try:
            smart196_index = title.index("smart_196_raw")
        except:
            smart196_index = -1
        try:
            smart197_index = title.index("smart_197_raw")
        except:
            smart197_index = -1
        try:
            smart198_index = title.index("smart_198_raw")
        except:
            smart198_index = -1
         
        for line_index in range(1,len(diskinfo)-1):
            now_line = list(diskinfo[line_index])
            model_df = 1
            '''
            if (model_index+1):
                model_type = re.findall(r'iqweijST(\d*)', now_line[model_index])
                if len(model_type) > 0:
                    if not model_type[0]:  
                        model_df = 1
                    else:
                        model_df = 0
            '''
            if model_df:
                row = []
                smart5 = try_int(now_line, smart5_index)
                smart9 = try_int(now_line, smart9_index)
                smart196 = try_int(now_line, smart196_index)
                smart197 = try_int(now_line, smart197_index)
                smart198 = try_int(now_line, smart198_index)
                smart198 = try_int(now_line, smart198_index)
                smart2 = try_int(now_line, smart2_index)
                smart3 = try_int(now_line, smart3_index)
                smart4 = try_int(now_line, smart4_index)
                smart193 = try_int(now_line, smart193_index)
                smart194 = try_int(now_line, smart194_index)
                smart195 = try_int(now_line, smart195_index)
                if (failure_index + 1):
                   if int(now_line[failure_index]) > 0:
                       row = [smart2, smart3, smart4, smart5, smart9, smart193, smart194, smart195, smart196, smart197, smart198, int(now_line[failure_index])*100, 0]
                       print(row)
                   else:
                       row = [smart2, smart3, smart4, smart5, smart9, smart193, smart194, smart195, smart196, smart197, smart198, 0, 100]
                   writer.writerow(row)                
     

if __name__ == '__main__':
    file_list = []
    os_csv_path("/ssd/diskcsv", file_list)
    fileListALL = create_fileprocess(file_list, processes_num)
    processes = list()
    #write_csv(fileListALL[0], 'fuck.csv')    
    for i in range(processes_num):
        
        #p = Process(target=csv_ana, args=(fileListALL[i], f_l, sige_all[i],i,))
        input_file = "12thread"+str(i)+".csv"
        p = Process(target=write_csv, args=(fileListALL[i], input_file,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


