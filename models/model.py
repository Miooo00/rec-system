import paddle
import paddle.nn.layer
from paddle import nn


# (30404, 144343, 1066, 2650)
hyper_parameters = {
    'em_1': 128,
    'em_2': 512,
    'em_3': 8,
    'em_4': 16,
    'hid_1': 64,
    'hid_2': 64,
    'hid_3': 64,
    'lr': 0.1,
    'batch_size': 8
}


class TFC(paddle.nn.Layer):
    def __init__(self, num_users, num_goods, num_cat, num_brandsn, batch_size):
        super(TFC, self).__init__()
        self.batchsize = batch_size
        self.user_em = nn.Embedding(num_users, 128) # 1,128
        self.good_em = nn.Embedding(num_goods, 512) # 1,512
        self.cat_em = nn.Embedding(num_cat, 8) # 1,8
        self.brand_em = nn.Embedding(num_brandsn, 16) # 1,16
        # 1,664
        self.Dnn = nn.Sequential(
            nn.Linear(678, 1024),
            nn.Linear(1024, 2048),
            nn.Linear(2048, 2560),
            nn.Linear(2560, 1280),
            nn.Linear(1280, 512),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
    def forward(self, input):
        # input (1, 11)
        user_coding = self.user_em(input[0])
        good_coding = self.good_em(input[1])
        cat_coding = self.cat_em(input[2])
        brand_coding = self.brand_em(input[3])
        users_actions = paddle.concat([input[4], input[5], input[6], input[7]], axis=1).reshape((self.batchsize, 1, -1))
        # print(user_coding, good_coding, cat_coding, brand_coding, users_actions)
        usergood_info = paddle.concat([user_coding, good_coding, cat_coding, brand_coding, users_actions], axis=2)
        res = self.Dnn(usergood_info)
        return res

    def predict(self, input):
        user_coding = self.user_em(input[0])
        good_coding = self.good_em(input[1])
        cat_coding = self.cat_em(input[2])
        brand_coding = self.brand_em(input[3])
        users_actions = paddle.concat([input[4], input[5], input[6], input[7]]).reshape((1, -1))
        # print(user_coding, good_coding, cat_coding, brand_coding, users_actions)
        usergood_info = paddle.concat([user_coding, good_coding, cat_coding, brand_coding, users_actions], axis=1)
        res = self.Dnn(usergood_info)
        return res


# def train(model, epoches, train_loader, y_true):
#     model.train()
#     optim = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
#     for epoch in range(epoches):
#         print(f'-----------------epoch:{epoch}-----------------')
#         for i in range(len(train_loader)):
#             predict = model(train_loader[i])
#             loss = nn.functional.cross_entropy(predict, y_true[i], axis=1, soft_label=True)
#             if i % 200 == 0:
#                 loss_ = paddle.to_tensor(loss, place='cpu')
#                 print(f'-----------------{i}-----------------')
#                 print(loss_)
#             loss.backward()
#             optim.step()
#             optim.clear_grad()

def train(model, epoches, train_loader):
    model.train()
    optim = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    for epoch in range(epoches):
        print(f'-----------------epoch:{epoch}-----------------')
        for batch_id, data in enumerate(train_loader()):
            x, y = data
            predict = model(x)
            loss = nn.functional.cross_entropy(predict, y, soft_label=True)
            if batch_id % 200 == 0:
                loss_ = paddle.to_tensor(loss, place='cpu')
                print(f'-----------------batch:{batch_id}-----------------')
                print(loss_)
            loss.backward()
            optim.step()
            optim.clear_grad()


# model = TFC(1,2,3)
# init_data = paddle.randn((10, 9))
# test = model(init_data)
# print(test)

# a = paddle.to_tensor([0])
# b = paddle.ones((1, 3), dtype='int64')
#
# print(a)
# test = nn.Embedding(100, 8)
# b = test(a)
# print(b)
