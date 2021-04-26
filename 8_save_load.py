# 保存和加载模型
import torch
from torchvision import models

# -----------------------------------------------------------------------------------------------------------------------
# 保存
# torch.save(model.state_dict(), PATH)
# 加载
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
# 在运行推理之前，务必调用 model.eval() 去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致 模型推断结果不一致。
# -----------------------------------------------------------------------------------------------------------------------
# 保存/加载完整模型
# torch.save(model, PATH)
# model = torch.load(PATH)
# model.eval()
# -----------------------------------------------------------------------------------------------------------------------
# 保存和加载 Checkpoint 用于推理/继续训练
# 保存
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             ...
#             }, PATH)
# 加载
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.eval()
# # - or -
# model.train()

