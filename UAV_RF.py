



import torch
import torch.nn as nn
import torchmetrics
from matplotlib import pyplot as plt
from torch.autograd import Variable










import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_dataset_path  =  'data/UAV/train/1_train/train_data.npy'
train_labels_path = 'data/UAV/train/1_train/train_label.npy'
train_dataset_path2  =  'data/UAV/train/2_train/ep_dataset.npy'
train_labels_path2 = 'data/UAV/train/2_train/ep_label.npy'
train_dataset_path3  =  'data/UAV/AL/AL1/AL1_data.npy'
train_labels_path3 = 'data/UAV/AL/AL1/AL1_label.npy'
train_dataset_path4  =  'data/UAV/water/patched_dataset.npy'
train_labels_path4 = 'data/UAV/water/patched_labels.npy'
train_dataset_path5  =  'data/all/21_all/21_alldata.npy'
train_labels_path5 = 'data/all/21_all/21_alllabel.npy'
train_dataset_path6  =  'data/all/20_sea/20_data.npy'
train_labels_path6 = 'data/all/20_sea/20_label.npy'
train_dataset_path7  =  'data/all/22_sea/22_data.npy'
train_labels_path7 = 'data/all/22_sea/22_label.npy'




val_dataset_path  =  'data/all/21beach/21_data.npy'
val_labels_path = 'data/all/21beach/21_label.npy'





classes = [
"_background_",
"Emergent plant",
"Vegetation",
"Pit and pond",
"Muddy beach"];

# 从numpy文件中加载数据
def load_npy(path):
    return np.load(path)

# 从numpy文件加载数据和标签
# 加载训练集
def load_train_data():
    train_dataset_rw = load_npy(train_dataset_path).astype('float32')
    train_dataset_rw2 = load_npy(train_dataset_path2).astype('float32')
    train_dataset_rw3 = load_npy(train_dataset_path3).astype('float32')
    train_dataset_rw4 = load_npy(train_dataset_path4).astype('float32')
    train_dataset_rw5 = load_npy(train_dataset_path5).astype('float32')
    train_dataset_rw6 = load_npy(train_dataset_path6).astype('float32')
    train_dataset_rw7 = load_npy(train_dataset_path7).astype('float32')
    train_dataset_rw1 = np.concatenate((train_dataset_rw, train_dataset_rw2), axis=0)
    train_dataset_rw2 = np.concatenate((train_dataset_rw1, train_dataset_rw3), axis=0)
    train_dataset_rw3 = np.concatenate((train_dataset_rw2, train_dataset_rw4), axis=0)
    train_dataset_rw4 = np.concatenate((train_dataset_rw3, train_dataset_rw5), axis=0)
    train_dataset_rw5 = np.concatenate((train_dataset_rw4, train_dataset_rw6), axis=0)
    train_dataset_rw = np.concatenate((train_dataset_rw5, train_dataset_rw7), axis=0)
    print('训练集数据形状：',train_dataset_rw.shape)
    train_target_rw = load_npy(train_labels_path).astype('float32')
    train_target_rw2 = load_npy(train_labels_path2).astype('float32')
    train_target_rw3 = load_npy(train_labels_path3).astype('float32')
    train_target_rw4 = load_npy(train_labels_path4).astype('float32')
    train_target_rw5 = load_npy(train_labels_path4).astype('float32')
    train_target_rw6 = load_npy(train_labels_path4).astype('float32')
    train_target_rw7 = load_npy(train_labels_path4).astype('float32')
    train_target_rw1 = np.concatenate((train_target_rw, train_target_rw2), axis=0)
    train_target_rw2 = np.concatenate((train_target_rw1, train_target_rw3), axis=0)
    train_target_rw3 = np.concatenate((train_target_rw2, train_target_rw4), axis=0)
    train_target_rw4 = np.concatenate((train_target_rw3, train_target_rw5), axis=0)
    train_target_rw5 = np.concatenate((train_target_rw4, train_target_rw6), axis=0)
    train_target_rw = np.concatenate((train_target_rw5, train_target_rw7), axis=0)
    print('训练集标签形状：',train_target_rw.shape)
    train_target_rw = torch.from_numpy(train_target_rw)#torch.Size([399, 256, 256, 11])
    return train_dataset_rw, train_target_rw

def load_val_data():
  val_dataset_rw = load_npy(val_dataset_path).astype('float32')
  print('验证集数据形状：',val_dataset_rw.shape)
#   val_dataset_rw = torch.from_numpy(val_dataset_rw)#torch.Size([399, 256, 256, 3])
  val_target_rw = load_npy(val_labels_path).astype('float32')
  print('验证集标签形状：',val_target_rw.shape)
  val_target_rw = torch.from_numpy(val_target_rw)#torch.Size([399, 256, 256, 11])
  return val_dataset_rw, val_target_rw

X_train,y_train = load_train_data()
X_val,y_val=load_val_data()



y_train = y_train.numpy()
y_val = y_val.numpy()
print(type(X_train))
print(type(y_train))
print(type(X_val))
print(type(y_val))
print()

X_train_index=[]
X_val_index=[]
# X_test_index=[]
for i in range(3953):
    if np.all(X_train[i] == 0) or np.all(y_train[i, 3, :, :] == 1):
        #
        X_train_index.append(i)
print('训练集中空的patch的数量：',len(X_train_index))

for j in range(399):
    if np.all(X_val[j] == 0) or np.all(y_val[j, 3:, :] == 1):
        #
        X_val_index.append(j)
print('验证集中空的patch的数量：',len(X_val_index))


X_train = np.delete(X_train, X_train_index,axis=0)
y_train = np.delete(y_train, X_train_index,axis=0)
X_val = np.delete(X_val, X_val_index,axis=0)
y_val = np.delete(y_val, X_val_index,axis=0)


y_train_class = np.argmax(y_train, axis=-1)
y_val_class = np.argmax(y_val, axis=-1)
print(X_train.shape)
print(y_train_class.shape)
print(X_val.shape)
print(y_val_class.shape)
print(np.max(X_train))
print(np.min(X_train))
print(np.max(y_train_class))
print(np.min(y_train_class))
print(np.max(X_val))
print(np.min(X_val))
print(np.max(y_val_class))
print(np.min(y_val_class))
del y_val,y_train


# 将输入特征和标签重塑为2D数组，以便于使用随机森林
num_samples_train, height_train, width_train, num_channels_train = X_train.shape
num_samples_val, height_val, width_val, num_channels_val = X_val.shape
X_train_reshaped = X_train.reshape(num_samples_train* height_train * width_train, num_channels_train)
y_train_reshaped = y_train_class.reshape(num_samples_train* height_train * width_train)
X_val_reshaped = X_val.reshape(num_samples_val* height_val * width_val, num_channels_val)
y_val_reshaped = y_val_class.reshape(num_samples_val* height_val * width_val)
del X_train,y_train_class,X_val,y_val_class
print(X_train_reshaped.shape)
print(y_train_reshaped.shape)
print(X_val_reshaped.shape)
print(y_val_reshaped.shape)

# # 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=80)
#
# 拟合随机森林分类器
rf.fit(X_train_reshaped, y_train_reshaped)



# 创建一个空的列表，用于存储每张图片的预测结果
predicted_labels_list = []



# 使用随机森林进行预测
predicted_labels_list = rf.predict(X_val_reshaped)


# 将预测结果列表转换为NumPy数组
predicted_labels_array = np.array(predicted_labels_list)
print(predicted_labels_array.shape)
print(y_val_reshaped.shape)


# 获取类别数量
num_classes = 5
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val_reshaped, predicted_labels_array)
tp = np.diag(cm)
tn = np.sum(cm) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - tp)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
recall = tp / (tp + fn)
acc=tp/(tp+fp)
print(recall)
print(acc)

# 设定标签类别
labels = ['Blackground', 'Emergent plant', 'Vegetation', 'Pit and pond','Muddy beach']  # 替换为您的实际类别标签

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# 设置图形属性
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# 显示混淆矩阵
plt.show()

