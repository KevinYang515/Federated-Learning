with open('./mypaper/data_distribution_3000_new_14.txt', 'r') as rf:
    device_information = rf.readlines()
    
with open('./mypaper/device_info_3000.txt', 'r') as rf:
    device_information_detail = rf.readlines()

# Read data from input file
def read_data():
    # It will preprocess the data format into the dictionary we want.
    Return the information of 

    data_distribution = {}
    each_class = []

    for i in range(10):
        each_class.append(0)
        
    for i in range(len(device_information)):
        temp = []
        temp_class = device_information[i].split(';')[1][1:-1].split(' ')
        temp_emd = device_information[i].split(';')[0]
        temp_info = device_information_detail[i][1:-2].split(',')
        temp_var = device_information[i].split(';')[2][:-1]
        
        for j in range(10):
            temp.append([each_class[j], each_class[j] + int(temp_class[j])])
            each_class[j] += int(temp_class[j])
            
        data_distribution[i] = {}
        data_distribution[i]['training time'] = int(temp_info[0])
        data_distribution[i]['transmission time'] = float(temp_info[1])
        data_distribution[i]['data_quantity'] = int(temp_info[2])
        data_distribution[i]['emd'] = float(temp_emd)
        data_distribution[i]['variance'] = float(temp_var)
        data_distribution[i]['data_distribution'] = temp

    return data_distribution