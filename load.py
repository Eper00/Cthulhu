import mujoco
import mujoco.viewer
import numpy as np
import time
model = mujoco.MjModel.from_xml_path('./assets/tentacle.xml')
data = mujoco.MjData(model)
tendon_sensor_ids = [model.sensor(name=f"tendon{i+1}_pos").id for i in range(3)]

with mujoco.viewer.launch_passive(model,data) as viwer:
    while viwer.is_running():
        data.ctrl[:]=[0.2,0.2,0.2]
        mujoco.mj_step(model,data)
        
        print(data.qpos)
      
        viwer.sync()

#mujoco.viewer.launch(model,data)

