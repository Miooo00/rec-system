import paddle


class MyDataset(paddle.io.Dataset):
    def __init__(self, data_path):
        # single data meg
        self.data_list = []
        # overall users
        self.user_set = set()
        # overall goods
        self.good_set = set()
        with open(data_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            for line in lines:
                items = line.split(',')
                single_data = []
                for item in items:
                    single_data.append(item.strip())
                self.user_set.add(items[0])
                self.good_set.add(items[1])
                self.data_list.append(single_data)

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


t = MyDataset('E:/Complex Designer 3/train/traindata_user/part-00000')
# print(t.data_list[1])
print(t[0])
train_loader = paddle.io.DataLoader(t, batch_size=8, shuffle=True, num_workers=0)
print(t.user_set)
print(len(t.user_set))
print(len(t.good_set))