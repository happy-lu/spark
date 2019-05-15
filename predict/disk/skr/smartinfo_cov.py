import numpy 
import os, sys, time
import math,copy
from  multiprocessing import Process
smartctl_flag = ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11', '12', '13', '15', '22', '183', '184', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '220', '222', '223', '224', '225', '226', '240', '241', '242', '250', '251', '252', '254', '255']

processes_num = 16

def os_csv_path(rootdir, file_list):
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
         path = os.path.join(rootdir,list[i])
         if os.path.isfile(path) and (".csv" in path):
             file_list.append(path)
         if os.path.isdir(path):
             os_csv_path(path, file_list)

def comp_fl_ifitdiff(l1, l2):
    if len(l1) == len(l2):
        for i in range(len(l1)):
            if str(l1[i]).encode('utf-8').decode('utf-8-sig') != str(l2[i]).encode('utf-8').decode('utf-8-sig'):
                return True
        return False
    else:
        return True

def find_smartctl_flag(l2):
    ret_flag_list = []
    for i in l2:
        if "raw" in str(i).encode('utf-8').decode('utf-8-sig'):
            lls = str(i).encode('utf-8').decode('utf-8-sig').split("_")
            ret_flag_list.append(lls)
    return ret_flag_list

def comp_model_ifST(line_in):
    if "ST" in str(line_in[2]).encode('utf-8').decode('utf-8-sig')[0:2] or "st" in str(line_in[2]).encode('utf-8').decode('utf-8-sig')[0:2]:
        return False  #True - ST model ; False - none ST model
    else:
        return True

def open_csv(path):
    first_line = numpy.loadtxt(path, str, delimiter= ',')
    return first_line

def csv_read(loadcsv_path):
    loadcsv = numpy.loadtxt(loadcsv_path, str, delimiter= ',')
    return loadcsv
    
            
def sige_pearson(line, list_all):
    #sige = [sige_failure, sige_failure_square, sige_smartctl_raw, sige_smartctl_raw_square, sige_times_value, sige_N]
    for smartctl_flag_raw in range(6,len(line) + 1,2):
        if line[smartctl_flag_raw] != '':
            locat = int((smartctl_flag_raw-6)/2)
            raw_value = int(line[smartctl_flag_raw])
            failure_value = int(line[4])
            list_all[0][locat] = failure_value + list_all[0][locat] #record failure of this flag
            list_all[1][locat] = failure_value + list_all[1][locat]
            list_all[2][locat] = raw_value + list_all[2][locat]
            list_all[3][locat] = raw_value*raw_value + list_all[3][locat]
            list_all[4][locat] = raw_value*failure_value + list_all[4][locat]
            list_all[5][locat] += 1
    
def corr(sige, smartctl_flag):
    ret_list = {}
    for tag in range(len(sige[5])):
        if sige[5][tag] != 0: 
            up_part = sige[4][tag] - sige[0][tag]*sige[2][tag]/sige[5][tag]
            down_part = (sige[1][tag] - sige[0][tag]*sige[0][tag]/sige[5][tag])*(sige[3][tag] - sige[2][tag]*sige[2][tag]/sige[5][tag])
            if down_part == 0:
                p = "Divide 0" 
            else:
                p = up_part / math.sqrt(down_part)
            ret_list.update({smartctl_flag[tag]:str(p)})
            #print(p)
        else:
            p = 0
            #print(p)
            ret_list.update({smartctl_flag[tag]:str(p)})
    file_name = "corr_result.txt"
    a = open(file_name, mode='w')
    a.write(str(ret_list))
    a.close()
    return ret_list

def csv_ana(file_list, f_l, sige, i):
    #print(file_list)
    for csv in file_list:
        print(csv)
        diskinfo = csv_read(csv)
        if not comp_fl_ifitdiff(f_l, diskinfo[0]):
            new_flag = find_smartctl_flag(diskinfo[0])
            for line_num in range(1,len(diskinfo)):
                if not comp_model_ifST(diskinfo[line_num]):
                    sige_pearson(diskinfo[line_num], sige)
    file_name = str(i)+".txt"
    a = open(file_name, mode='w')
    for i in sige:
        for j in range(len(i)): 
            a.write(str(i[j]))
            if j != (len(i)-1):
                a.write(",")
        a.write('\n')
    a.close()
    print(sige)

def sigeall_2_sige(sige, process_num):
    sige_all = create_sige_all(sige, process_num)
    for j in range(int(process_num)):
        file_name = str(j)+".txt"
        a = numpy.loadtxt(file_name, delimiter= ',')
        for i in range(len(a)):
            sige_all[j][i] = a[i]
    for i in range(len(sige[0])):
        for j in range(int(process_num)):
            sige[0][i] = int(sige_all[j][0][i]) + int(sige[0][i])
            sige[1][i] = int(sige_all[j][1][i]) + int(sige[1][i])
            sige[2][i] = int(sige_all[j][2][i]) + int(sige[2][i])
            sige[3][i] = int(sige_all[j][3][i]) + int(sige[3][i])
            sige[4][i] = int(sige_all[j][4][i]) + int(sige[4][i])
            sige[5][i] = int(sige_all[j][5][i]) + int(sige[5][i])

def sige_test():
    sige = create_sige(smartctl_flag)
    sigeall_2_sige(sige, processes_num)
    a = corr(sige, smartctl_flag)
    print(a)

def filelist_test(num):
    file_list = []
    os_csv_path("/ssd/diskcsv", file_list)
    all_li = create_fileprocess(file_list, num)
    for i in all_li:
        for j in i:
            print(j)

def create_sige_all(sige, process_num):
    sige_all = list()
    for i in range(int(process_num)):
        sige_model_new = copy.deepcopy(sige)
        sige_all.append(sige_model_new)
    return sige_all


def create_sige(smartctl_flag):
    sige_failure = [0]*len(smartctl_flag)
    sige_smartctl_raw = [0]*len(smartctl_flag)
    sige_times_value = [0]*len(smartctl_flag)
    sige_failure_square = [0]*len(smartctl_flag)
    sige_smartctl_raw_square = [0]*len(smartctl_flag)
    sige_N = [0]*len(smartctl_flag)
    sige = [sige_failure, sige_failure_square, sige_smartctl_raw, sige_smartctl_raw_square, sige_times_value, sige_N]
    return sige

def create_fileprocess(file_list, proce_num):
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

if __name__ == '__main__': 
    file_list = []
    pxy={}
    ####
    sige = create_sige(smartctl_flag)
    sige_all = create_sige_all(sige, processes_num)
    ####
    f_l = open_csv("/home/skrdata/pycsv/line.csv")
    os_csv_path("/ssd/diskcsv", file_list)
    fileListALL = create_fileprocess(file_list, processes_num)
    processes = list()
    #print("1000")
    #time.sleep(1000)
    for i in range(processes_num):
        p = Process(target=csv_ana, args=(fileListALL[i], f_l, sige_all[i],i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    #print(sige_all)
    sigeall_2_sige(sige, processes_num)
    corr(sige, smartctl_flag)       
