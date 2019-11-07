import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import deepmind_lab
from . import LEVELS, MAP
import time
import json

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
            "{}. (HINT: you can install Atari dependencies by running "
            "'pip install gym[atari]'.)".format(e))


class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, scene, colors = 'RGB_INTERLEAVED', width = 84, height = 84, **kwargs):
        super(DeepmindLabEnv, self).__init__(**kwargs)

        if not scene in LEVELS:
            raise Exception('Scene %s not supported' % (scene))

        self._colors = colors
        self._lab = deepmind_lab.Lab(scene, [self._colors, 'INSTR'], \
            dict(fps = str(60), width = str(width), height = str(height)))

        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), dtype = np.uint8)

        self._last_observation = None
        self.total_reward = 0.0
        self.len = 0
        self.start = time.clock()

        # self.ale = atari_py.ALEInterface()

    def done(self, obs):
        instr = obs['INSTR']
        if instr:
            instr = json.loads(instr)
            for command_idx in range(1, instr['nCommands'] + 1):
                command = instr['Command' + str(command_idx)]['Command']
                if command == "EpisodeFinished":
                    return True
        return False

    def step(self, action):
        if not self._lab.is_running():
            infos = {'episode': {'r': self.total_reward,
                                 'l': self.len,
                                 't': time.clock() - self.start}}
            return self._last_observation, 0.0, False, infos
        reward = self._lab.step(ACTION_LIST[action], num_steps=1)
        obs = self._lab.observations()[self._colors]
        self._last_observation = obs if obs is not None else np.copy(self._last_observation)
        done = self.done(self._lab.observations())
        if done:
            infos = {'episode': {'r': self.total_reward,
                                 'l': self.len,
                                 't': time.clock() - self.start}}
            self.total_reward = 0.0
            self.len = 0
            self.start = time.clock()
        else:
            self.len += 1
            self.total_reward += reward
            infos = dict()
        return self._last_observation, reward, done, infos

    def reset(self):
        self._lab.reset()
        self._last_observation = self._lab.observations()[self._colors]
        self.start = time.clock()
        self.total_reward = 0.0
        self.len = 0
        return self._last_observation

    def seed(self, seed = None):
        self._lab.reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)


    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._lab.observations()[self._colors]
        #elif mode is 'human':
        #   pop up a window and render
        else:
            super(DeepmindLabEnv, self).render(mode=mode) # just raise an exception

    def get_action_meanings(self):
        return english_names_of_actions


def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTION_LIST = [
    _action(0, 0, 0, 0, 0, 0, 0),  # noop
    _action(-20,   0,  0,  0, 0, 0, 0), # look_left
    _action( 20,   0,  0,  0, 0, 0, 0), # look_right
    #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
    #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
    # _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    # _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(  0,   0,  0,  1, 0, 0, 0), # forward
    # _action(  0,   0,  0, -1, 0, 0, 0), # backward
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
]

english_names_of_actions = [
    'NOOP',
    'look_left',
    'look_right',
    # 'look_up',
    # 'look_down',
    # 'strafe_left',
    # 'strafe_right',
    'forward',
    # 'backward',
    # 'fire',
    # 'jump',
    # 'crouch',
]