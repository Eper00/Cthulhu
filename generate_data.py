import mujoco
import numpy as np
from parameters import path_xml_toy
def get_model_and_data(path_xml):
    model = mujoco.MjModel.from_xml_path(path_xml)
    data = mujoco.MjData(model)
    return model, data
def simulation_random_trajectories(model,data,siumaltion_time):
    print(model.jnt_range)
    U=[]
    Qpos=[]
    Qvel=[]
    u=np.random.uniform(model.jnt_range[:,0],model.jnt_range[:,1],model.nu)
    U.append(u)
    
model,data=get_model_and_data(path_xml_toy)
simulation_random_trajectories(model,data,10)
