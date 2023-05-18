import paddle


class MyDataset(paddle.io.Dataset):
    def __init__(self, data, labels):
        self.data_list = data
        self.labels = labels
    def __getitem__(self, index):
        return self.data_list[index], self.labels[index]

    def __len__(self):
        return len(self.data_list)


