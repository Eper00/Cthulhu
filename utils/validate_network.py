from networks import DynamicsModel
import torch
from utils.support import load_model
from utils.parameters import path,validation_steps
import numpy as np
import mujoco

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DynamicsModel()
model.load_state_dict(torch.load('./models/f.pt', map_location=device))
model.to(device)
model.eval()

sim, data = load_model(path)

errors = []

for _ in range(validation_steps):

    x = np.concatenate((data.qpos, data.qvel))
    x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    u = np.random.uniform(0.12, 0.34, 3)
    data.ctrl[:] = u
    u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).to(device)

    # 🔥 NN predikció (dt nélkül!)
    with torch.no_grad():
        x_pred = model.step(x_torch, u_torch)

    x_pred = x_pred.cpu().numpy()[0]

    # valódi MuJoCo step
    mujoco.mj_step(sim, data)
    x_real = np.concatenate((data.qpos, data.qvel))

    err = np.linalg.norm(x_pred - x_real)
    errors.append(err)

print("Mean 1-step error:", np.mean(errors))