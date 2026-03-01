import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.parameters import nq, nv, epochs, batch_size, lr


class DynamicsModel(nn.Module):
    def __init__(self, state_dim=nq+nv, input_dim=3, hidden_dim=512):
        super().__init__()

        self.state_dim = state_dim
        self.input_dim = input_dim

        # --- háló ---
        self.fc1 = nn.Linear(state_dim + input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

        # --- normalizáció (buffer -> elmentődik state_dict-be) ---
        self.register_buffer("x_mean", torch.zeros(state_dim))
        self.register_buffer("x_std", torch.ones(state_dim))

        self.register_buffer("u_mean", torch.zeros(input_dim))
        self.register_buffer("u_std", torch.ones(input_dim))

        self._init_weights()

    # -------------------------------------------------
    # Súly inicializálás (stabilabb: Xavier)
    # -------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------
    # Normalizáció beállítása
    # -------------------------------------------------
    def set_normalization(self, x_data, u_data):
        self.x_mean.copy_(x_data.mean(0))
        self.x_std.copy_(x_data.std(0) + 1e-6)

        self.u_mean.copy_(u_data.mean(0))
        self.u_std.copy_(u_data.std(0) + 1e-6)

    # -------------------------------------------------
    # Forward: Δx predikció
    # -------------------------------------------------
    def forward(self, x, u):

        x_norm = (x - self.x_mean) / self.x_std
        u_norm = (u - self.u_mean) / self.u_std

        xu = torch.cat([x_norm, u_norm], dim=-1)

        h = torch.tanh(self.fc1(xu))
        h = torch.tanh(self.fc2(h))

        delta_norm = self.fc3(h)

        # visszaskálázás
        delta = delta_norm * self.x_std

        return delta

    # -------------------------------------------------
    # Diszkrét lépés
    # -------------------------------------------------
    def step(self, x, u):
        return x + self.forward(x, u)


# =====================================================
# TRAIN
# =====================================================

def train_dynamics(
    model,
    inputs,
    states,
    next_states,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    states = torch.tensor(states, dtype=torch.float32)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # 🔥 Δx target
    delta = next_states - states

    model.to(device)

    # 🔥 normalizáció CPU-n számolva (stabilabb)
    model.set_normalization(states, inputs)

    dataset = torch.utils.data.TensorDataset(states, inputs, delta)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):

        total_loss = 0

        for x, u, target in loader:

            x = x.to(device)
            u = u.to(device)
            target = target.to(device)

            pred = model(x, u)

            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {_+1} | Loss: {total_loss/len(loader):.6f}")

    print("Training finished.")
    torch.save(model.state_dict(), './models/f.pt')


# =====================================================
# MULTI-STEP LOSS
# =====================================================

def multi_step_loss(model, x_seq, u_seq):

    B, H, _ = u_seq.shape

    x_pred = x_seq[:, 0]
    loss = 0

    for t in range(H):
        x_pred = model.step(x_pred, u_seq[:, t])
        loss += F.mse_loss(x_pred, x_seq[:, t+1])

    return loss / H