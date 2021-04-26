# 关于pytorch的自动求导以及反向传递
import torch

# ------------------------------------------------------------------------------
# 创建一个张量，设置requires_grad=True 来跟踪与它相关的计算
x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
# ------------------------------------------------------------------------------
# 针对张量做一个操作
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
print(y.grad_fn)
# <AddBackward0 object at 0x00000163FAAC2748>
z = y**2*3
out = z.mean()
print(z, out)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
# tensor(27., grad_fn=<MeanBackward0>)
out.backward()  # 反向传播
print(x.grad)
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])
# ------------------------------------------------------------------------------
# 雅可比向量积
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:  # 求距离 sqrt(square(x1) + square(x2) + ... + square(n))
    y = y * 2
print(y)
# tensor([1210.8331,   32.9330, -407.0987], grad_fn=<MulBackward0>)
# 现在在这种情况下，y 不再是一个标量。torch.autograd 不能够直接计算整个雅可比，
# 但是如果我们只想要雅可比向量积，只需要简单的传递向量给 backward 作为参数。
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
# tensor([5.1200e+01, 5.1200e+02, 5.1200e-02])
# ------------------------------------------------------------------------------
# 将代码包裹在 with torch.no_grad()，来停止对从跟踪历史中的 .requires_grad=True 的张量自动求导。
print(x.requires_grad)  # True
print((x**2).requires_grad)  # True
with torch.no_grad():
    print((x**2).requires_grad)  # False
# ------------------------------------------------------------------------------
