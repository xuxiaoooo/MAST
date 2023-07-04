import scipy.io
import pandas as pd

# 加载 .mat 文件
mat = scipy.io.loadmat('/Users/xuxiao/WorkBench/AMA_EEG/data/features/PSDdata/IPS_3_PSD.mat')
data_list = mat['None']

# 在列表中遍历每个元组
for data_tuple in data_list:
    # 如果元组的第三个元素（索引为2）是 'table'
    if data_tuple[2] == b'table':
        # 打印出元组的第四个元素（索引为3），这应该是实际的数据
        print(data_tuple[3])

# # 将 numpy 数组转换为 pandas DataFrame
# df = pd.DataFrame(data)

# # 将 DataFrame 保存为 excel 文件
# df.to_excel('/Users/xuxiao/WorkBench/AMA_EEG/data/features/PSDdata/IPS_3_PSD.xlsx', index=False)
