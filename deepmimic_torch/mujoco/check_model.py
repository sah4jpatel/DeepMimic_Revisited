#%%
import mujoco
from mujoco import MjData, MjModel
import mujoco_viewer

#%%


if __name__ == "__main__":
    env_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/envs/dp_env_v2.xml'
    env_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/humanoid/humanoid.xml'
    model = MjModel.from_xml_path(env_file_path)

    print("nq (generalized coordinates):", model.nq)
    print("nv (velocity coordinates):", model.nv)
    print("nu (number of actuators):", model.nu)
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("ngeom:", model.ngeom)

    data = mujoco.MjData(model)
    print(data.qpos)
    print(data.qvel)

    # for body_id in range(model.nbody):
    #     name_start = model.name_bodyadr[body_id]
    #     name = ""
    #     while model.names[name_start] != '\x00':
    #         name += model.names[name_start]
    #         name_start += 1
    #     print(f"Body {body_id} name: {name}")


# %%


env_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/envs/dp_env_v2.xml'
env_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/humanoid/humanoid.xml'
model = MjModel.from_xml_path(env_file_path)

print("nq (generalized coordinates):", model.nq)
print("nv (velocity coordinates):", model.nv)
print("nu (number of actuators):", model.nu)
print("nbody:", model.nbody)
print("njnt:", model.njnt)
print("ngeom:", model.ngeom)

data = mujoco.MjData(model)
print(data.qpos)
print(data.qvel)
# %%
