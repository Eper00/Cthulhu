import mujoco
import mujoco.viewer
import numpy as np
import time

# Model betöltése
model = mujoco.MjModel.from_xml_path('./assets/tentacle.xml')
data = mujoco.MjData(model)

# Maximumok tárolása
num_actuators = 3
max_forces = np.zeros(num_actuators)
min_lengths = np.full(num_actuators, np.inf)
max_lengths = np.zeros(num_actuators)

# Sensor ID-k a tenderekhez
tendon_sensor_ids = [model.sensor(name=f"tendon{i+1}_pos").id for i in range(3)]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Vezérlőjelek random generálása a megadott tartományban
        data.ctrl[:num_actuators] = np.random.uniform(0.12, 0.34, num_actuators)

        # Szimuláció lépés
        mujoco.mj_step(model, data)

        # Tendon erők lekérdezése
        tendon_forces = data.actuator_force[:num_actuators]
        max_forces = np.maximum(max_forces, np.abs(tendon_forces))

        # Tendon hosszok lekérdezése
        tendon_lengths = np.array([data.sensordata[i] for i in tendon_sensor_ids])
        min_lengths = np.minimum(min_lengths, tendon_lengths)
        max_lengths = np.maximum(max_lengths, tendon_lengths)

        # Viewer frissítése
        viewer.sync()

# Eredmények kiírása
for i in range(num_actuators):
    print(f"Actuator {i+1}:")
    print(f"  Max erő: {max_forces[i]:.3f} N")
    print(f"  Min hossz: {min_lengths[i]:.4f} m")
    print(f"  Max hossz: {max_lengths[i]:.4f} m")
    print(f"  Max hossz változás: {max_lengths[i] - min_lengths[i]:.4f} m")