import mujoco
import numpy as np
import pandas as pd
from parameters import num_actuators, sample_size, path,nq,nv
from support import load_model

model, data = load_model(path)



# --- Header generálás ---
state_columns = (
    [f"qpos_{i}" for i in range(nq)] +
    [f"qvel_{i}" for i in range(nv)]
)

next_state_columns = [f"{name}_next" for name in state_columns]

input_columns = [f"u_{i}" for i in range(num_actuators)]

output_columns = (
    [f"tendon_{i}" for i in range(3)] +
    ["tip_x", "tip_y", "tip_z"]
)

# --- Storage ---
states = []
next_states = []
inputs = []
outputs = []

tendon_sensor_ids = [model.sensor(f"tendon{i+1}_pos").id for i in range(3)]
tip_sensor_id = model.sensor("tip_pos").id

for _ in range(sample_size):

    # 1️⃣ State mérés
    state = np.concatenate((data.qpos, data.qvel))

    tendon_lengths = np.array(
        [data.sensordata[i] for i in tendon_sensor_ids]
    )

    tip_pos = data.sensordata[
        model.sensor_adr[tip_sensor_id]:
        model.sensor_adr[tip_sensor_id] + 3
    ]

    # 2️⃣ Input generálás
    u = np.random.uniform(0.12, 0.34, num_actuators)
    data.ctrl[:num_actuators] = u

    # 3️⃣ Step
    mujoco.mj_step(model, data)

    # 4️⃣ Next state
    next_state = np.concatenate((data.qpos, data.qvel))

    # 5️⃣ Tárolás
    states.append(state.copy())
    inputs.append(u.copy())
    next_states.append(next_state.copy())
    outputs.append(np.concatenate((tendon_lengths, tip_pos)))

# --- Mentés ---
pd.DataFrame(states, columns=state_columns)\
    .to_csv("./data/states.csv", index=False)

pd.DataFrame(inputs, columns=input_columns)\
    .to_csv("./data/inputs.csv", index=False)

pd.DataFrame(next_states, columns=next_state_columns)\
    .to_csv("./data/next_states.csv", index=False)

pd.DataFrame(outputs, columns=output_columns)\
    .to_csv("./data/outputs.csv", index=False)

print("Dataset successfully saved.")