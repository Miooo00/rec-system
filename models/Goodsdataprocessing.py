import os
import numpy as np

train_goodsdata_path = 'E:/Complex Designer 3/train/traindata_goodsid'
train_goodsdata_files = os.listdir(train_goodsdata_path)


def get_data(data_files, root):
    data_list = []
    for str_file in data_files:
        print(str_file)
        data_path = root + f'/{str_file}'
        with open(data_path, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                for line in lines:
                    items = line.split(',')
                    single_data = []
                    for item in items:
                        single_data.append(item.strip())
                    data_list.append(single_data)
    print('--------------goods dict create!---------------')
    return data_list

# goods_data = get_data(train_goodsdata_files, train_goodsdata_path)
"""
为每个商品创建一个字典，由字典映射该商品的品类和品牌信息
该步骤将商品的有限信息加入(concat)到原始数据集中
"""
def get_goods_dict():
    goods_data = get_data(train_goodsdata_files, train_goodsdata_path)
    goods_dict = {}
    for a_good in goods_data:
        goods_dict[a_good[0]] = a_good[1:]
    return goods_dict

# goods_dict = get_goods_info(goods_data)
# print(goods_dict)

def binary_encode(num, max_len):
    t_col = []
    enc_info = np.zeros(max_len, dtype='int64')
    while num/2 != 0:
        t_col.append(num % 2)
        num = int(num/2)
    for i in range(len(t_col)):
        enc_info[i] = t_col[i]
    return enc_info

