import paddle
import paddle.nn.layer
from paddle import nn


class TFC(paddle.nn.Layer):
    def __init__(self, num_users, num_goods, embedding_size):
        super(TFC, self).__init__()
        self.num_users = num_users
        self.num_goods = num_goods
        self.em_size = embedding_size

        self.layer = paddle.nn.Sequential(
            nn.Linear(9, 100),
            nn.Linear(100, 20),
            nn.Linear(20, 30),
            nn.Flatten()
        )

    def forward(self, input):
        res = self.layer(input)
        return res


model = TFC(1,2,3)
init_data = paddle.randn((10, 9))
test = model(init_data)
print(test)