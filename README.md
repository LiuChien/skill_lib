# skill_lib


## Example
### env_wrapper.py
```python
# demo code for SkillWrapper in env_wrapper.py
import gym
from env_wrapper import SkillWrapper

SKILLS = [[2,2,2,2],[3,3,3,3],[4,4,4],[5,5,5]]
...
env = gym.make("Alien-ram-v0")
env = SkillWrapper(env, SKILLS)
...
```

### mamager
```python
# demo code for AtariPolicyManager in manager.py
from env_wrapper import SkillWrapper
from manager import AtariPolicyManager
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
...
env = gym.make("Alien-ram-v0")
atari_manager = AtariPolicyManager(env=env, model=TRPO, policy=MlpPolicy, save_path = "/path/to/store/location", verbose=1)
...
skills= [[2,2,2,2],[3,3,3,3],[4,4,4],[5,5,5]]
episode_ave_reward, action_ave_reward = atari_manager.get_rewards(skills)

```
