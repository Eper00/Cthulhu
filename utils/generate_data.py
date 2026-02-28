import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from parameters import num_actuators, sample_size, path
from support import load_model

model, data = load_model(path)
input_columns = ["u_0", "u_1", "u_2"]
state_columns = ["l_0", "l_1", "l_2", "t_x", "t_y", "t_z"]
inputs = []
states = []
next_states = []

tendon_sensor_ids = [model.sensor(f"tendon{i+1}_pos").id for i in range(3)]
tip_sensor_id = model.sensor("tip_pos").id

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(sample_size):

        # 1️⃣ Aktuális state mérése
        tendon_lengths = np.array([data.sensordata[i] for i in tendon_sensor_ids])
        tip_pos = data.sensordata[
            model.sensor_adr[tip_sensor_id]:
            model.sensor_adr[tip_sensor_id] + 3
        ]
        state = np.concatenate((tendon_lengths, tip_pos))

        # 2️⃣ Input generálás
        u = np.random.uniform(0.12, 0.34, num_actuators)
        data.ctrl[:num_actuators] = u

        # 3️⃣ Egy step
        mujoco.mj_step(model, data)

        # 4️⃣ Next state
        tendon_lengths_next = np.array([data.sensordata[i] for i in tendon_sensor_ids])
        tip_pos_next = data.sensordata[
            model.sensor_adr[tip_sensor_id]:
            model.sensor_adr[tip_sensor_id] + 3
        ]
        next_state = np.concatenate((tendon_lengths_next, tip_pos_next))

        # 5️⃣ Tárolás
        states.append(state.copy())
        inputs.append(u.copy())
        next_states.append(next_state.copy())

        viewer.sync()

# Mentés
pd.DataFrame(states,columns=state_columns).to_csv("./data/states.csv", index=False)
pd.DataFrame(inputs,columns=input_columns).to_csv("./data/inputs.csv", index=False)
pd.DataFrame(next_states,columns=state_columns).to_csv("./data/next_states.csv", index=False)

