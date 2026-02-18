import torch

ckpt = torch.load("ik_model.pth", weights_only=False)

print("x_min =", ckpt["x_scaler_min"])
print("x_scale =", ckpt["x_scaler_scale"])
print("y_min =", ckpt["y_scaler_min"])
print("y_scale =", ckpt["y_scaler_scale"])
