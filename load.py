import mujoco
import mujoco.viewer
import numpy as np
import time
model = mujoco.MjModel.from_xml_path('./assets/tentacle.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model,data) as viwer:
    while viwer.is_running():
        data.ctrl[:]=np.random.uniform(0.12,0.34,model.nu)
        mujoco.mj_step(model,data)
        
        viwer.sync()
        time.sleep(0.01)        