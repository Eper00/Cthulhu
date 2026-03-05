import mujoco
import mujoco.viewer
import numpy as np
import time
from parameters import path_xml_tentacle,path_xml_toy
path_xml=path_xml_toy
model = mujoco.MjModel.from_xml_path(path_xml)
data = mujoco.MjData(model)
# beállítjuk a home pozíciót
'''
data.qpos[:] = [0]*model.nq  # vagy a keyframe qpos értékei
data.ctrl[:] = [0.34]*model.nu  # ha ctrl is kell
'''
mujoco.viewer.launch(model,data)