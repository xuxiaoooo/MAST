import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from collections import Counter

def findmiss():
    base_folder_path = '/home/user/xuxiao/MAST/data/band_mat'
    folder_list = ['IPS_3','IPS_6','IPS_10','IPS_15','IPS_24','IPS_30']
    id_sets = []
    for folder in folder_list:
        folder_path = os.path.join(base_folder_path, folder)
        ids = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('.mat'):
                    id = filename[-8:-4]
                    ids.append(id)
        id_sets.append(set(ids))
    common_ids = set.intersection(*id_sets)
    all_ids = set.union(*id_sets)
    missing_ids = all_ids - common_ids
    return list(common_ids), list(missing_ids)

def concatness(ls):
    bands = ['alpha', 'beta', 'delta', 'theta', 'whole_band']
    ips = ['3', '6', '10', '15', '24', '30']
    ls = [str(x) for x in ls]
    for band in bands:
        data_dict = {}
        for ip in ips:
            for id in ls:
                file = f'/home/user/xuxiao/MAST/data/band_mat/IPS_{ip}/IPS_{ip}_{band}_{id}.mat'
                print(file)
                if os.path.exists(file):
                    mat = loadmat(file)
                    data = mat['EEG']['data'][0][0]
                    print(data.shape)
                    data_json = json.dumps(data.tolist())
                    if id in data_dict:
                        data_dict[id].append(data_json)
                    else:
                        data_dict[id] = [data_json]
        df = pd.DataFrame.from_dict(data_dict, orient='index', columns=ips)
        df.to_csv(f'/home/user/xuxiao/MAST/data/band_csv/{band}.csv', index_label='id')

def merge_all(ls):
    bands = ['delta', 'theta', 'alpha', 'beta']
    ips = ['3', '6', '10', '15', '24', '30']
    ls = [str(x) for x in ls]
    data_dict = {}
    for band in bands:
        for ip in ips:
            for id in ls:
                file = f'/home/user/xuxiao/MAST/data/band_mat/IPS_{ip}/IPS_{ip}_{band}_{id}.mat'
                print(file)
                if os.path.exists(file):
                    mat = loadmat(file)
                    data = mat['EEG']['data'][0][0]
                    if data.shape[1] < 2000:
                        data = np.pad(data, ((0,0), (0,2000 - data.shape[1])), 'constant', constant_values=(0))
                    if id in data_dict:
                        data_dict[id].append(data)
                    else:
                        data_dict[id] = [data]
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['B1Co1', 'B1Co2', 'B1Co3', 'B1Co4', 'B1Co5', 'B1Co6',
                                                                    'B2Co1', 'B2Co2', 'B2Co3', 'B2Co4', 'B2Co5', 'B2Co6',
                                                                    'B3Co1', 'B3Co2', 'B3Co3', 'B3Co4', 'B3Co5', 'B3Co6',
                                                                    'B4Co1', 'B4Co2', 'B4Co3', 'B4Co4', 'B4Co5', 'B4Co6',])
    df = df.reset_index().rename(columns={"index": "id"})
    df.to_pickle(f'/home/user/xuxiao/MAST/data/band_csv/all.pkl')

    

if __name__ == '__main__':
    ls, ls2 = findmiss()
    # concatness(ls)
    merge_all(ls)