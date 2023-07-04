import pickle
import matplotlib.pyplot as plt
import numpy as np

file_paths = [
    "/Users/xuxiao/WorkBench/AMA_EEG/code/draw/roc_alpha_BD-MDD.pkl",
    "/Users/xuxiao/WorkBench/AMA_EEG/code/draw/roc_beta_BD-MDD.pkl",
    "/Users/xuxiao/WorkBench/AMA_EEG/code/draw/roc_delta_BD-MDD.pkl",
    "/Users/xuxiao/WorkBench/AMA_EEG/code/draw/roc_theta_BD-MDD.pkl",
    "/Users/xuxiao/WorkBench/AMA_EEG/code/draw/roc_whole_band_BD-MDD.pkl"
]

roc_curves = []
for file_path in file_paths:
    with open(file_path, "rb") as f:
        roc_curves.append(pickle.load(f))

labels = [
    "alpha",
    "beta",
    "delta",
    "theta",
    "whole band",
]

# 设置字体
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 18,
        }
plt.rc('font', **font)

# 画ROC曲线
plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 45度线
for i, (fpr, tpr) in enumerate(roc_curves):
    plt.plot(fpr, tpr, lw=2, linestyle='--', label=labels[i])

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16, fontname="Arial")
plt.ylabel('True Positive Rate', fontsize=16, fontname="Arial")
plt.title('Receiver Operating Characteristic Curves', fontsize=20)
plt.legend(loc="lower right", prop={'family':'Arial', 'size':16})
plt.show()
