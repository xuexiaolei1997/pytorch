# pytorch关于GPU的使用
# model = nn.DataParallel(model) 两个GPU或以上使用
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------------------
input_size, output_size = 5, 2
batch_size, data_size = 30, 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------------------
# 生成一个玩具数据
class RandomDataset(Dataset):
    def __init__(self, size, length):
        super(RandomDataset, self).__init__()
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# ------------------------------------------------------------------------------
# 造数据
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
# ------------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        print('\tIn Model: input size', x.size(), 'output size', output.size())
        return output


# ------------------------------------------------------------------------------
# 创建模型并且数据并行处理
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:  # 两个GPU以上
    print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(), "output_size", output.size())
