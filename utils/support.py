import mujoco
import pandas as pd
def load_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model,data
def load_data():
    inputs=pd.read_csv('./data/inputs.csv').to_numpy()
    outputs=pd.read_csv('./data/outputs.csv').to_numpy()
    states=pd.read_csv('./data/states.csv').to_numpy()
    next_states=pd.read_csv('./data/next_states.csv').to_numpy()
    return inputs,states,next_states,outputs
