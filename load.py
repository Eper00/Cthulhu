from utils.support import load_data
from utils.parameters import dt
from networks import train_dynamics,DynamicsModel
inputs,states,next_states,outputs=load_data()
dynamics_model=DynamicsModel()
train_dynamics(dynamics_model,inputs,states,next_states)

#mujoco.viewer.launch(model,data)

