# 如何构建一个网络
# 主要学习，构建网络，给定多少卷积层，以及应该如何去做，优化器使用等
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# ------------------------------------------------------------------------------


net = Net()
print(net)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

# ------------------------------------------------------------------------------
# 一个模型可训练的参数可以通过调用 net.parameters() 返回
params = list(net.parameters())
print(len(params))  # 10
print(params[0].size())  # torch.Size([6, 1, 5, 5])
# ------------------------------------------------------------------------------
# 尝试随机生成一个 32x32 的输入。注意：期望的输入维度是 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# tensor([[ 0.0044, -0.0180, -0.1060, -0.1041,  0.0020,  0.0399, -0.0769,  0.1196,
#          -0.0488, -0.0692]], grad_fn=<AddmmBackward>)
# 把所有参数梯度缓存器置零，用随机的梯度来反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))
# ------------------------------------------------------------------------------
# 定义损失函数
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)  # tensor(0.5590, grad_fn=<MseLossBackward>)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# <MseLossBackward object at 0x0000029F4F6F0C88>
# <AddmmBackward object at 0x0000029F4D4DBE08>
# <AccumulateGrad object at 0x0000029F4F6F0C88>
# ------------------------------------------------------------------------------
# 反向传播
net.zero_grad()     # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# conv1.bias.grad before backward
# tensor([0., 0., 0., 0., 0., 0.])
# conv1.bias.grad after backward
# tensor([-0.0129,  0.0032, -0.0148, -0.0179, -0.0078,  0.0094])
# ------------------------------------------------------------------------------
# 更新神经网络参数  weight = weigh - learning_rate * gradient
learning_rate = .01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
# ------------------------------------------------------------------------------
# 使用不同的更新规则，类似于 SGD, Nesterov-SGD, Adam, RMSProp, 等
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=.01)
# in your training loop
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update
# ------------------------------------------------------------------------------
