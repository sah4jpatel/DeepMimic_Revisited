from environment import DeepMimicGymEnv
from drl import DRL

# env.
# env.play_trajectory()
# env = DeepMimicGymEnv()
# obs, reward, done = env.step(None)

drl = DRL(load_models=True)
for i in range(1000):
    print("epoch: " + str(i))
    drl.rollout(i, False)
    drl.update()
drl.agent.save_models()

for i in range(2):
    drl.rollout(i, True)