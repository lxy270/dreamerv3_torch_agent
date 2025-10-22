import os
os.environ["MUJOCO_GL"] = "egl"

import gym
import json
import math
import heapq
import random
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
from tqdm import tqdm
import memory_maze

def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())




# seq gen for panda
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

task = 'Maze' # 'Pong-v5' # "spin"
act_mode = "expert" # ['zero', 'random', 'expert']
train_param = [200, 1000, '/scorpio/home/yubei-stu-2/dreamerv3_torch_ver/data/' + f"seq-{task}-{act_mode}.npz"]
test_param = [200, 100, '/scorpio/home/yubei-stu-2/dreamerv3_torch_ver/data/' + f"seq-{task}-{act_mode}-test.npz"]
keys = ["obs", "reward", "done", "action"]
obs_key = ['agent_pos', 'agent_dir']

from dreamer import Dreamer, make_env
import tools
import argparse
from ruamel.yaml import YAML
yaml = YAML(typ='safe', pure=True)
import torch
from parallel import Damy

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

def load_dreamer(task, ckpt_path, device):
    print("load_dreamer")
    config_path = '/scorpio/home/yubei-stu-2/agent_dreamerv3_torch/configs.yaml' # server
    with open(config_path, 'r') as file:
        config = yaml.load(file)
    defaults = {}
    for name in ['defaults', 'memorymaze']: # trigger, plus re.match(cnn_key)
        recursive_update(defaults, config[name])
    
    config = argparse.Namespace(**defaults)
    config.device = device
    config.task = task
    env = Damy(make_env(config, "train", 0))
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0] # trigger

    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        None, # no logger, change __init__
        None,
    )
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(ckpt_path,weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"],strict=False)
    # tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    agent._should_pretrain._once = False
    return agent, env



# env = gym.make('MemoryMaze-9x9-ExtraObs-v0')
B = 25
envs = [gym.make('MemoryMaze-9x9-ExtraObs-v0') for _ in range(B)]

def obs_dreamer(inp_obs, done, is_first):
    obs = inp_obs.copy()
    obs["is_first"]    = np.array(is_first, dtype=bool)
    obs["is_last"]     = np.array(done, dtype=bool)
    obs["is_terminal"] = np.zeros_like(obs["is_last"])
    # for k in obs.keys():
    #     obs[k] = obs[k][None, :]
    return obs

def seq_gen(seq_cnt, seq_len, savepath, agent, file_id=None):
    if file_id is not None:
        savepath = savepath.removesuffix('.npz') + f'{file_id}' + '.npz'
    print('plan save to: ', savepath)
    min_seq = seq_len
    data = {k: [] for k in keys}
    while len(data['obs']) < seq_cnt:
        print(len(data['obs']))
        seq = {k: [] for k in keys}
        obs =[env.reset() for env in envs]
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        agent_state = None
        done = np.zeros(B, dtype=bool)    # (B,)
        is_first = np.ones(B, dtype=bool) # (B,)
        agent_obs = obs_dreamer(obs, done, is_first)
        for _ in range(seq_len):
            action, agent_state = agent(agent_obs, done, agent_state, training=False)
            action = action['action'].argmax(axis=-1).cpu().numpy().astype(np.int64)
            result = [env.step(act) for env, act in zip(envs, action)]
            obs, reward, done = zip(*[p[:3] for p in result])
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
            #obs, reward, done, info = env.step(action) # a_t -> o_t
            agent_obs = obs_dreamer(obs, done, np.zeros(B, dtype=bool)) # add dim after append
            obs = np.concatenate([obs[k] for k in obs_key], axis=-1)
            reward = np.array(reward)
            done = np.array(done, dtype=bool)
            # print(done, reward)
            # print('obs', obs.shape) # [B, 4]
            # print('reward', reward) # [B, 1]
            seq['obs'].append(obs)
            seq['reward'].append(reward)
            seq['done'].append(done)
            seq['action'].append(action)
            if done.any():
                print(done)
                break
        if len(seq['obs']) < seq_len: # 0.6 * seq_len:
            print('skip ', len(seq['obs']))
            continue

        for k in keys:
            seq[k] = np.swapaxes(np.array(seq[k]), 0, 1) # [t, b, d] -> [b, t, d]
            for i in range(B):
                data[k].append(seq[k][i])

    data = {k: np.array(data[k]) for k in keys}
    for key in data.keys():
        print(key, data[key].shape, np.isinf(data[key]).any(), np.isnan(data[key]).any())
    metadata = {}
    
    np.savez_compressed(savepath, **data, metadata=metadata)
    print('saved ', savepath)



if __name__ == "__main__":
    ckpt_path = '/scorpio/home/yubei-stu-2/agent_dreamerv3_torch/agent_maze2.pt'
    task = 'memorymaze_9x9'
    agent, env = load_dreamer(task, ckpt_path, device='cuda:0')
    # seq_gen(*test_param, agent=agent)
    for file_id in range(10):
        seq_gen(*train_param, agent=agent, file_id=file_id)