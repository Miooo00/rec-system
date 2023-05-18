import pandas as pd
import paddle
from paddle import nn
import numpy as np
from models.MyDataset import MyDataset
from models.model import TFC, train


def binary_encode(num, max_len):
    t_col = []
    enc_info = np.zeros(max_len, dtype='int64')
    while num/2 != 0:
        t_col.append(num % 2)
        num = int(num/2)
    for i in range(len(t_col)):
        enc_info[i] = t_col[i]
    return enc_info


labels = ['user_id', 'goods_id', 'cat_id', 'brandsn', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt', 'target']
dataset_path = 'test.csv'


df = pd.read_csv(dataset_path)
# 舍列
df.drop(columns=['expose_start_time', 'dt'], inplace=True)
# shuffle
df = df.sample(frac=1)
# 标签
targets = df['target'][:50000].tolist()
target = []
for t in targets:
    if t == 0:
        target.append([[0, 1]])
    else:
        target.append([[1, 0]])

target = paddle.to_tensor(np.array(target, dtype='float32'))
print(target)
# targets = paddle.to_tensor(np.array(df['target'][:100]).reshape((-1, 1)), dtype='float32')
# print(targets)
# 先取前10列测试
data_test = df[:50000]
# 转换为numpy列表
data_list = np.array(df[:50000])
# [数据集 [tensor_1, tensor_2 ....]]
# print(data_test)
# print(data_list)
max_ = [4, 2, 5, 3]


dataset = []
for item in data_list:
    cur_data = []
    p = 0
    for i in item[:4]:
        cur_data.append(paddle.to_tensor([i]))
    for i in item[4:-1]:
        cur_data.append(paddle.to_tensor(binary_encode(i, max_[p]), dtype='float32'))
        p += 1
    dataset.append(cur_data)

data = MyDataset(dataset, target)
batchsize = 128
user_size = 30404
good_size = 144343
cat_size = 1066
brandsn_size = 2650


model = TFC(user_size, good_size, cat_size, brandsn_size, batchsize)
# res = model(data[0])
x, y = data[0]
# print(x)
# print(y)
train_loader = paddle.io.DataLoader(
    data,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0,
    drop_last=True)

train(model, 5, train_loader)
pre = model.predict(x)
print(pre)
# for batch_id, data in enumerate(train_loader()):
#     print(batch_id, data)


