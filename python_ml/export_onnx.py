import torch
import torch.nn as nn

class IKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )

    def forward(self,x):
        return self.net(x)

# Load trained model
checkpoint = torch.load("ik_model.pth", weights_only=False)
model = IKNet()
model.load_state_dict(checkpoint["model"])
model.eval()

dummy_input = torch.randn(1,3)

torch.onnx.export(
    model,
    dummy_input,
    "ik_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=9
)

print("ONNX export complete")
