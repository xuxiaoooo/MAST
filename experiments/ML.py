import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

band = 'alpha'
crowd = 'IPS_3'

df = pd.read_excel('/Users/xuxiao/WorkBench/AMA_EEG/data/features/PSDdata/IPS_3_PSD.xlsx')
pd_label = pd.read_excel('/Users/xuxiao/WorkBench/AMA_EEG/data/label/suicide.xlsx')
df['patient_IDs'] = df['patient_IDs'].astype(str)
pd_label['简编'] = pd_label['简编'].astype(str)
df = df.merge(pd_label[['简编', '诊断']], left_on='patient_IDs', right_on='简编', how='left')
df['诊断'] = df['诊断'].replace({2: 0, 4: 1}) # 将 '诊断' 列的 2 和 4 分别替换为 0 和 1
df = df.rename(columns={'诊断': 'label'}).drop(columns=['patient_IDs','简编']).dropna()
print(df.columns)

# 定义分类器
classifiers = {
    "SVC": {"model": SVC(probability=True),
        "params": {'clf__C': [0.1, 1], 'clf__kernel': ['linear', 'rbf']}},
    
    "RandomForest": {"model": RandomForestClassifier(),
                    "params": {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 5]}},
    
    # "LightGBM": {"model": LGBMClassifier(),
    #             "params": {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.1, 0.01]}},
    
    # "XGBoost": {"model": XGBClassifier(),
    #             "params": {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.1, 0.01]}}
}

mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, (classifier_name, classifier_info) in enumerate(classifiers.items()):
    print("正在训练分类器:", classifier_name)
    classifier = classifier_info["model"]
    params = classifier_info["params"]
    result_metrics = []  # 用于存储评估指标的结果
    y_tests = []
    y_preds = []
    tprs = []
    aucs = []
    confusion_mat_sum = np.zeros((2,2)) # 对于二分类问题
    
    # 训练模型并评估
    X = df.drop(columns=['label'])
    y = df['label']
    print(y.value_counts())

    cv = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        pipe = Pipeline([('scl', StandardScaler()), ('clf', classifier)])
        gs = GridSearchCV(estimator=pipe, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        y_proba = gs.predict_proba(X_test)[:, 1]
        
        # 计算ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        # 计算混淆矩阵
        confusion_mat = confusion_matrix(y_test, y_pred)
        confusion_mat_sum += confusion_mat

        result_metrics.append([accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)])
        y_tests.extend(y_test)
        y_preds.extend(y_pred)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(f'/Users/xuxiao/WorkBench/AMA_EEG/code/draw/{crowd}_{band}_{classifier_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 显示混淆矩阵
    print("Confusion matrix:\n", confusion_mat_sum)
    cm = confusion_matrix(y_tests, y_preds)
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')
    plt.savefig(f'/Users/xuxiao/WorkBench/AMA_EEG/code/draw/{crowd}_{band}_{classifier_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    result_metrics = np.array(result_metrics)
    print("准确率: {:.3f}".format(result_metrics[:,0].mean()))
    print("精确率: {:.3f}".format(result_metrics[:,1].mean()))
    print("召回率: {:.3f}".format(result_metrics[:,2].mean()))
    print("F1得分: {:.3f}".format(result_metrics[:,3].mean()))