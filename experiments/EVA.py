import torch, json, pickle
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import torch_geometric.nn as geo_nn
from torch_geometric.utils import dense_to_sparse
import sys
sys.path.append('../utils/')
from MAST import MAST

class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.stack([torch.tensor(feature[j], dtype=torch.float32) for j in range(len(feature))]) for feature in data]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].to(device), self.labels[idx].to(device)
    
def permutation_test(actual, predicted, num_permutations=1000):
    mse_observed = ((actual - predicted) ** 2).mean()
    mse_permuted = []
    for _ in range(num_permutations):
        predicted_permuted = np.random.permutation(predicted)
        mse_permuted.append(((actual - predicted_permuted) ** 2).mean())
    p_value = np.mean([mse > mse_observed for mse in mse_permuted])
    return p_value

crowd = 'effect'

df = pd.read_pickle(f'/home/user/xuxiao/MAST/data/band_csv/all.pkl')
# 将标签转换为数字
pd_label = pd.read_excel('/home/user/xuxiao/MAST/data/lists.xlsx')
df['id'] = df['id'].astype(str)
pd_label['简编_x'] = pd_label['简编_x'].astype(str)
df = df.merge(pd_label[['简编_x', '性别（0男1女）_x']], left_on='id', right_on='简编_x', how='left')
df = df.rename(columns={'性别（0男1女）_x': 'label'})
df = df.drop(columns=['id', '简编_x']).dropna()
# 降采样平衡
class_counts = df['label'].value_counts()
min_class_count = class_counts.min()
df = df.groupby('label').apply(lambda x: x.sample(min_class_count)).reset_index(drop=True)
print(df)

device = torch.device("cuda")
model = MAST().to(device)

df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'])

train_dataset = SequenceDataset(df_train[df.columns.drop(["label"])].values.tolist(), df_train['label'].values)
test_dataset = SequenceDataset(df_test[df.columns.drop(["label"])].values.tolist(), df_test['label'].values)

train_dataloader = DataLoader(train_dataset, batch_size=2)
test_dataloader = DataLoader(test_dataset, batch_size=2)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

patience = 20
epochs_no_improve = 0
min_val_loss = float('inf')

train_losses = []
val_losses = []
acc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
conf_matrices = []
roc_aucs = []
roc_curves = []

for epoch in range(30):
    total_loss = 0
    for i, (data, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        labels = labels.float()
        loss = loss_function(output, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        for data, labels in test_dataloader:
            output = model(data)
            labels = labels.float()
            val_loss = loss_function(output, labels)
            total_val_loss += val_loss.item()

            probabilities = torch.sigmoid(output).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())

        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        # 计算各项指标
        acc = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        conf_matrices.append(conf_matrix)

        # 计算 ROC 曲线和 AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        roc_curves.append((fpr, tpr))

    model.train()
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")
    
# 训练结束后，计算并打印总的指标
print("Final Metrics:")
print(f"Accuracy: {np.mean(acc_scores)}")
print(f"F1 Score: {np.mean(f1_scores)}")
print(f"Precision: {np.mean(precision_scores)}")
print(f"Recall: {np.mean(recall_scores)}")

all_labels_np = np.array(all_labels)
all_predictions_np = np.array(all_predictions)
mse = ((all_labels_np - all_predictions_np) ** 2).mean()
p_value = permutation_test(all_labels_np, all_predictions_np)
print(f"MSE: {mse}")
print(f"Permutation test p-value: {p_value}")

# 绘制训练和验证损失曲线
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.savefig(f'/home/user/xuxiao/MAST/draw/loss_{crowd}.jpg', dpi=300, bbox_inches='tight')

# 绘制最后一个 epoch 的混淆矩阵
plt.matshow(conf_matrices[-1], cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrices[-1].shape[0]):
    for j in range(conf_matrices[-1].shape[1]):
        plt.text(x=j, y=i, s=conf_matrices[-1][i, j], va='center', ha='center')
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig(f'/home/user/xuxiao/MAST/draw/confusion_matrix_{crowd}.jpg', dpi=300, bbox_inches='tight')

# 绘制最后一个 epoch 的 ROC 曲线
with open(f'/home/user/xuxiao/MAST/draw/roc_{crowd}.pkl', 'wb') as f:
    pickle.dump(roc_curves[-1], f)
plt.figure()
plt.plot(*roc_curves[-1], color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig(f'/home/user/xuxiao/MAST/draw/roc_{crowd}.jpg', dpi=300, bbox_inches='tight')