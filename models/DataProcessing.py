import os



train_usersdata_path = 'E:/Complex Designer 3/train/traindata_user'
train_goodsdata_path = 'E:/Complex Designer 3/train/traindata_goodsid'
train_usersdata_files = os.listdir(train_usersdata_path)[0:1]
train_goodsdata_files = os.listdir(train_goodsdata_path)[0:1]


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
    return data_list

train_userdata = get_data(train_usersdata_files, train_usersdata_path)
users_info = {}
goods_info = {}
# train_goodsdata = get_data(train_goodsdata_files, train_goodsdata_path)
"""
1、设想合并每个用户对相同物品的行为,由于物品的浏览记录是按时间排序的,故相同的商品会合并到最后一次的请求中
2、由于目前对所有用户对该商品的偏好情况未知,现假设如下情况表示用户对商品存在兴趣：
    (1)对同一商品浏览记录超过1    说明：该条件不一定合适，后续根据情况调整
    (2)有收藏记录
    (3)已经加购或者购买过
3、对于请求时间和日期信息暂时不考虑

    综上暂时将用户id，商品id以及用户行为数据以及用户行为数据推导出的用户偏好作为训练集

"""
for data in train_userdata:
    if data[0] not in users_info:
        users_info[data[0]] = {data[1]: data[2:]}
    else:
        if data[1] not in users_info[data[0]]:
            users_info[data[0]][data[1]] = data[2:]
        else:
            users_info[data[0]][data[1]][0] = str(int(users_info[data[0]][data[1]][0]) + int(data[2]))
            users_info[data[0]][data[1]][1] = str(int(users_info[data[0]][data[1]][1]) + int(data[3]))
            users_info[data[0]][data[1]][2] = str(int(users_info[data[0]][data[1]][2]) + int(data[4]))
            users_info[data[0]][data[1]][3] = str(int(users_info[data[0]][data[1]][3]) + int(data[5]))
    if data[0] == '51804006b6d8110a7e121ca3e1e5fcf5':
        print(data)
print("division---------")

for k, v in users_info.items():
    for m, n in v.items():
        if int(n[0]) > 1 or int(n[1]) != 0 or int(n[2]) != 0 or int(n[3]) != 0:
            n.append('1')
        else:
            n.append('0')
info = users_info['51804006b6d8110a7e121ca3e1e5fcf5']
for k, v in info.items():
    print(k, v)

for k, v in users_info.items():
    for m, n in v.items():
        print(m, n)
# print(users_info['51804006b6d8110a7e121ca3e1e5fcf5'])

"""
# 筛选有购买行为的用户

users_set = set()
probablity_pos = []
probablity_neg = []

for data in train_userdata:
    if int(data[4]) != 0 or int(data[5]) != 0:
        probablity_pos.append(data)
        users_set.add(data[0])
    else:
        probablity_neg.append(data)

print(len(probablity_pos))
print(len(probablity_neg))
print(len(users_set))

with open('users_has_purchase.txt', 'w') as fp:
    res_str = ''
    for one_user in users_set:
        res_str += one_user + '\n'
    fp.write(res_str)
"""

"""
# 在5000名用户中过滤掉未有过购买行为的用户id
filter_set = set()
with open('processing/users_has_purchase.txt', 'r') as fp:
    content = fp.readlines()
    for line in content:
        filter_set.add(line)

pre_haspurchase = set()
with open('processing/uid.txt', 'r') as fp1:
    content = fp1.readlines()
    for line in content:
        if line in filter_set:
            pre_haspurchase.add(line)


print(len(pre_haspurchase))
with open('processing/filter_uid.txt', 'w') as fp2:
    res_str = ''
    for item in pre_haspurchase:
        res_str += item
    fp2.write(res_str)
"""