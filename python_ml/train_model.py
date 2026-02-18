import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[["x","y","z"]].values
Y = data[["t1","t2","t3"]].values

# Normalize
x_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaler = MinMaxScaler(feature_range=(-1,1))

X = x_scaler.fit_transform(X)
Y = y_scaler.fit_transform(Y)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2
)

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

model = IKNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for e in range(epochs):
    pred = model(X_train)
    loss = criterion(pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e+1)%10==0:
        print(f"Epoch {e+1}, Loss: {loss.item():.6f}")

torch.save({
    "model": model.state_dict(),
    "x_scaler_min": x_scaler.min_,
    "x_scaler_scale": x_scaler.scale_,
    "y_scaler_min": y_scaler.min_,
    "y_scaler_scale": y_scaler.scale_
}, "ik_model.pth")

print("Training complete")
