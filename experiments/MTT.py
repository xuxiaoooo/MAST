# Mat To Table
import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from collections import Counter

def findmiss():
    folder_path = '/Users/xuxiao/WorkBench/AMA_EEG/data/BD'
    numbers = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.mat'):
                number = filename[-8:-4]
                numbers.append(number)
    res_id_list = []
    counter = Counter(numbers)
    for number, count in counter.items():
        if count < 30:
            print(f'Number {number}: {count} times')
        else:
            res_id_list.append(number)
    return res_id_list

def concatness(ls):
    bands = ['alpha', 'beta', 'delta', 'theta', 'whole_band']
    ips = ['3', '6', '10', '15', '24', '30']
    for band in bands:
        data_dict = {}
        for ip in ips:
            for id in ls:
                mat = loadmat(f'/Users/xuxiao/WorkBench/AMA_EEG/data/BD/RESULTS_IPS_{ip}/IPS_{ip}_{band}/IPS_{ip}_{band}_{id}.mat')
                data = mat['EEG_filtered']['data'][0][0]
                print(data.shape)
                data_json = json.dumps(data.tolist())
                if id in data_dict:
                    data_dict[id].append(data_json)
                else:
                    data_dict[id] = [data_json]
        df = pd.DataFrame.from_dict(data_dict, orient='index', columns=ips)
        df.to_csv(f'/Users/xuxiao/WorkBench/AMA_EEG/data/{band}.csv', index_label='id')

if __name__ == '__main__':
    ls = findmiss()
    concatness(ls)