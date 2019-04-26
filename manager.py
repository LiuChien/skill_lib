import gym
import time
from env_wrapper import SkillWrapper
from collections import deque, OrderedDict
import os
import datetime
import yaml
import shutil
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
import glob
import time
#TODO: Define abstract class for policy manager
# provide logger with skills used frequency
#
class PolicyManager(object):
    
    def get_rewards():
        raise NotImplementedError





class AtariPolicyManager(object):
    def __init__(self, env, model, policy, save_path, preserve_model=10, verbose=0):
        """
        :param env: (gym.core.Env) gym env with discrete action space
        :param model: any model in stable_baselines e.g PPO2, TRPO...
        :param policy: (ActorCriticPolicy) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
        :param save_path: (str) path to store model and log
        :param preserve_model: (int) how much history model will be preserved 
        :param verbose: (int) 0,1 wheather or not to print the training process
        """
        # super(AtariPolicyManager, self).__init__()
        self.env = env
        self.model = model
        self.policy = policy
        self.preserve_model = preserve_model
        self.verbose = verbose
        self._save_model_name=deque()
        self._serial_num = 1
        if save_path is None:
            self.save_path = None
        elif os.path.exists(save_path):
            while(True):
                i = input("save path already exist: {}\n new name(n)/keep(k)/exit(e)?".format(save_path))
                if i=="n":
                    dir_name = input("new dir name: ")
                    new_path = os.path.abspath(os.path.join(save_path, dir_name))
                    os.makedirs(new_path)
                    self.save_path = new_path
                    break
                elif i=="k":
                    self.save_path = os.path.abspath(save_path)
                    break
                elif i=="R":
                    shutil.rmtree(save_path)
                    os.makedirs(save_path)
                    self.save_path = os.path.abspath(save_path)
                    break
                elif i=="e":
                    exit(0)
        else:
            self.save_path = save_path
            os.makedirs(save_path)
    def get_rewards(self, skills=[], train_total_timesteps=100000, eval_times=100, eval_max_steps=1000, model_save_name=None, log_action_skill=True):
    # def get_rewards(self, skills=[], train_total_timesteps=10, eval_times=10, eval_max_steps=10, model_save_name=None, log_action_skill=True):

        """
        
        :param skills: (list) the availiable action sequence for agent 
        e.g [[0,2,2],[0,1,1]]
        :param train_total_timesteps: (int)total_timesteps to train 
        :param eval_times: (int)the evaluation times
        e.g eval_times=100, evalulate the policy by averageing the reward of 100 episode
        :param eval_max_steps: (int)maximum timesteps per episode when evaluate
        :param model_save_name: (str)specify the name of saved model (should not repeat)
        :param log_action_skill: ()
        """

        env = SkillWrapper(self.env, skills=skills)
        env = DummyVecEnv([lambda: env])
        model = self.model(self.policy, env, verbose=self.verbose)
        
        strat_time = time.time()
        print("start to train agent...")
        model.learn(total_timesteps=train_total_timesteps)
        print("Finish train agent")

        if self.save_path is not None:
            if self.preserve_model>0:
                self.save_model(model, model_save_name, skills=skills)
        
        #TODO evaluate
        #eval model
        info = OrderedDict()
        if log_action_skill:
            action_statistic = OrderedDict()
            for i in range(env.action_space.n):
                action_statistic[str(env.action_space[i])]=0


        ep_reward = []
        ep_ave_reward = []
        print("start to eval agent...")
        for i in range(eval_times):
            obs = env.reset()
            total_reward = []
            for i in range(eval_max_steps):
                action, _states = model.predict(obs)
                obs, rewards, dones, info_ = env.step(action)
                total_reward.append(rewards[0])

                if log_action_skill is True:
                    action_statistic[str(env.action_space[action[0]])] = action_statistic[str(env.action_space[action[0]])] + 1

                if bool(dones[0]) is True:
                    break
            
            ep_reward.append(sum(total_reward))
            ep_ave_reward.append(sum(total_reward)/len(total_reward))
        
        
        print("Finish eval agent")
        print("Elapsed: {} sec".format(round(time.time()-strat_time, 3)))
        ave_score = sum(ep_reward)/len(ep_reward)
        ave_action_reward = sum(ep_ave_reward)/len(ep_ave_reward)
        ave_score_std = round(np.std(np.array(ep_reward)),3)

        # info.update({"ave_score":ave_score, "ave_score_std":ave_score_std, "ave_reward":ave_reward})
        info["ave_score"] = ave_score
        info["ave_score_std"] = ave_score_std
        info["ave_action_reward"] = ave_action_reward
        if log_action_skill:
            info.update(action_statistic)
        env.close()    
        
        #log result
        self.log(info)

        self._serial_num = self._serial_num + 1
        return ave_score, ave_action_reward
    
    def save_model(self, model, name=None, **kwargs):
        if name is None:
            name = "model_" + str(self._serial_num)
        
        save_name = os.path.join(self.save_path, name)
        if os.path.isfile(save_name+".pkl"):
            print("Warning: overwrite model: {}".format(save_name+".pkl"))
        model.save(save_name)

        if len(kwargs)!=0:
            with open('{}.yml'.format(save_name), 'w') as outfile:
                yaml.dump(kwargs, outfile, default_flow_style=False)

        
        if name not in self._save_model_name:
            self._save_model_name.append(name)

        if len(self._save_model_name) > self.preserve_model:
            remove_name = self._save_model_name.popleft()
            remove_name = remove_name + ".*"
            for rm_name in glob.glob(remove_name):
                os.remove(rm_name)
        
    def log(self, info=None):
        if info is not None:
            filename = os.path.join(self.save_path, "log.txt")
            # assert all(key in info for key in ["ave_reward", "ave_score"])
            with open(filename, 'a') as f:
                # if "ave_total_scroe" in info:
                print("{s:{c}^{n}}".format(s=(" Episode: " + str(self._serial_num) + " "), c='*', n=27), file=f)
                keys = info.keys()
                # keys.sort()
                for key in keys:
                    print("{}: {}".format(key, info[key]), file=f)
                print("{s:{c}^{n}}".format(s="", c='*', n=27), file=f)




                





        

        

