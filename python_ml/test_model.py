import torch
import numpy as np
from train_model import IKNet

checkpoint = torch.load("ik_model.pth", weights_only=False)

model = IKNet()
model.load_state_dict(checkpoint["model"])
model.eval()

x = np.array([[0.3,0.2,0.4]])

x_scaled = (x - checkpoint["x_scaler_min"]) * checkpoint["x_scaler_scale"]

with torch.no_grad():
    out = model(torch.tensor(x_scaled, dtype=torch.float32))
    y = out.numpy()

y_real = y / checkpoint["y_scaler_scale"] + checkpoint["y_scaler_min"]

print("Predicted Joint Angles:", y_real)
