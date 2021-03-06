import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import deepmind_lab
from . import LEVELS, MAP
import time
import datetime
import json
import csv
import os

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install Atari dependencies by running "
        "'pip install gym[atari]'.)".format(e))


def time_in_seconds(time):
    return (int(time) % 3600) % 60


class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, scene, colors='RGB_INTERLEAVED', width=84, height=84, **kwargs):
        super(DeepmindLabEnv, self).__init__(**kwargs)

        if not scene in LEVELS:
            raise Exception('Scene %s not supported' % (scene))

        self._colors = colors
        self._lab = deepmind_lab.Lab(scene, [self._colors, 'INSTR'],
                                     dict(fps=str(60), width=str(width), height=str(height)))

        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)

        self._last_observation = None
        self.total_reward = 0.0
        self.len = 0
        self.start = time.clock()
        self.report_path = None
        self.report_rank = 0

        self.sound_on = None
        self.distractor_on = None
        self.distractor_start_time = 0
        self.distractor_stop_time = -5
        self.rat_left_base_during_reward_time = False
        self.rat_left_during_distractor = False
        self.episode = 0
        self.position = "base1"
        self.missed_counter = 0
        self.early_counter = 0
        self.late_counter = 0
        self.correct_counter = 0
        self.distractor_counter = 0
        self.correct_distractor_counter = 0

        if "Sound" in scene or "sound" in scene:
            self.write_to_file = self.write_to_file_sound
            self.process_command = self.process_command_sound
        elif "nose" in scene or "Nose" in scene:
            self.write_to_file = self.write_to_file_nose_poke
            self.process_command = self.process_command_nose_poke
        elif "Memory" in scene or "memory" in scene:
            self.process_command = self.process_command_memory
            self.write_to_file = self.write_to_file_memory


        # self.ale = atari_py.ALEInterface()

    def set_report_path(self, path, rank):
        self.report_path = path
        self.report_rank = rank
        if not os.path.exists(self.report_path):
            os.mkdir(self.report_path)

    def write_to_file_sound(self, type_event, seconds):
        csv_name = str(self.report_path) + '/rat' + str(self.report_rank) + '_' + str(self.episode) + '.csv'
        if not os.path.exists(csv_name):
            with open(csv_name, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['time_stamp', 'event', 'missed', 'early', 'late', 'distracted', 'dist.OK', 'correct'])
        with open(csv_name, 'a') as fd:
            writer = csv.writer(fd)
            seconds = int(seconds)
            h = str(seconds // 3600)
            m = str((seconds % 3600) // 60)
            s = str((seconds % 3600) % 60)
            writer.writerow([str(self.episode) + "_" + h + ":" + m + ":" + s,
                             type_event, self.missed_counter, self.early_counter, self.late_counter,
                             self.distractor_counter, self.correct_distractor_counter, self.correct_counter])

    def process_command_sound(self, obs):
        if self.report_path is None:
            return
        instr = obs['INSTR']
        if instr:
            instr = json.loads(instr)
            for command_idx in range(1, instr['nCommands'] + 1):
                command = instr['Command' + str(command_idx)]['Command']
                if command == "Position":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = instr['Command' + str(command_idx)]['Opt']['Num2']
                    new_position = instr['Command' + str(command_idx)]['Opt']['String1']

                    if self.position == "base1" and new_position == "corridor":
                        self.write_to_file("not_in_base", time)
                        if self.sound_on:
                            self.rat_left_base_during_reward_time = True
                            self.write_to_file("rat_left_base_during_reward_time", time)
                        elif self.distractor_on or \
                                (self.distractor_stop_time - self.distractor_start_time < 5
                                 and self.distractor_stop_time >= self.distractor_start_time):
                            self.distractor_counter += 1
                            self.rat_left_during_distractor = True
                            self.write_to_file("rat_left_during_distractor", time)
                            self.distractor_stop_time = (self.distractor_start_time + 5) % 60
                        else:
                            self.early_counter += 1
                            self.write_to_file("left_early", time)
                    if self.position == "corridor" and new_position == "base1":
                        self.write_to_file("in_base", time)
                    self.position = new_position

                elif command == "Pickup":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = instr['Command' + str(command_idx)]['Opt']['Num2']
                    name = instr['Command' + str(command_idx)]['Opt']['String1']
                    self.correct_counter += 1
                    self.late_counter -= 1
                    self.write_to_file("correct_trial", time)

                elif command == "Timeout":
                    time = instr['Command' + str(command_idx)]['Opt']['Num1']

                elif command == "IndicationStatus":
                    self.episode = instr['Command' + str(command_idx)]['Opt']['Num1']
                    time = instr['Command' + str(command_idx)]['Opt']['Num2']
                    status = instr['Command' + str(command_idx)]['Opt']['String1']
                    if status == "sound_on":
                        self.sound_on = True
                        self.write_to_file("reward_time_started", time)
                    elif status == "sound_off":
                        self.sound_on = False
                        if self.rat_left_base_during_reward_time:
                            self.late_counter += 1
                            self.rat_left_base_during_reward_time = False
                            self.write_to_file("correct_or_late", time)
                        elif self.position == "base1":  # and not self.rat_left_base_during_reward_time:
                            self.missed_counter += 1
                            self.write_to_file("missed_trial", time)

                    elif status == "distractor_on":
                        self.distractor_on = True
                        self.rat_left_during_distractor = False
                        self.distractor_start_time = time_in_seconds(time)
                        self.write_to_file("distractor_time_started", time)
                    elif status == "distractor_off":
                        self.distractor_on = False
                        self.distractor_stop_time = time_in_seconds(time)
                        if not self.rat_left_during_distractor and self.distractor_stop_time - self.distractor_start_time >= 5:
                            self.correct_distractor_counter += 1
                            self.write_to_file("distractor_avoided", time)

    def write_to_file_nose_poke(self, type_event, seconds):
        csv_name = str(self.report_path) + '/rat' + str(self.report_rank) + '_' + str(self.episode) + '.csv'
        if not os.path.exists(csv_name):
            with open(csv_name, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['time_stamp', 'event', 'missed', 'early', 'distracted', 'dist.OK', 'correct'])
        with open(csv_name, 'a') as fd:
            writer = csv.writer(fd)
            seconds = int(seconds)
            h = str(seconds // 3600)
            m = str((seconds % 3600) // 60)
            s = str((seconds % 3600) % 60)
            writer.writerow([str(self.episode) + "_" + h + ":" + m + ":" + s,
                             type_event, self.missed_counter, self.early_counter,
                             self.distractor_counter, self.correct_distractor_counter, self.correct_counter])

    def process_command_nose_poke(self, obs):
        if self.report_path is None:
            return
        instr = obs['INSTR']
        if instr:
            instr = json.loads(instr)
            for command_idx in range(1, instr['nCommands'] + 1):
                command = instr['Command' + str(command_idx)]['Command']
                if command == "Position":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = instr['Command' + str(command_idx)]['Opt']['Num2']
                    new_position = instr['Command' + str(command_idx)]['Opt']['String1']

                    if self.position == "base1" and new_position == "nose_poke":
                        self.write_to_file("not_in_base", time)
                        if self.sound_on:
                            self.rat_left_base_during_reward_time = True
                            self.write_to_file("rat_left_base_during_reward_time", time)
                        elif self.distractor_on:
                            self.distractor_counter += 1
                            self.rat_left_during_distractor = True
                            self.write_to_file("rat_left_during_distractor", time)
                        else:
                            self.early_counter += 1
                            self.write_to_file("left_early", time)
                    if self.position == "nose_poke" and new_position == "base1":
                        self.write_to_file("in_base", time)
                    self.position = new_position

                elif command == "Pickup":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = instr['Command' + str(command_idx)]['Opt']['Num2']
                    name = instr['Command' + str(command_idx)]['Opt']['String1']
                    self.correct_counter += 1
                    self.write_to_file("correct_trial", time)

                elif command == "Timeout":
                    time = instr['Command' + str(command_idx)]['Opt']['Num1']

                elif command == "IndicationStatus":
                    self.episode = instr['Command' + str(command_idx)]['Opt']['Num1']
                    time = instr['Command' + str(command_idx)]['Opt']['Num2']
                    status = instr['Command' + str(command_idx)]['Opt']['String1']
                    if status == "sound_on":
                        self.sound_on = True
                        self.rat_left_base_during_reward_time = False
                        self.reward_start_time = time_in_seconds(time)
                        self.write_to_file("reward_time_started", time)
                    elif status == "sound_off":
                        self.sound_on = False
                        self.reward_stop_time = time_in_seconds(time)
                        if self.rat_left_base_during_reward_time:
                            self.rat_left_base_during_reward_time = False
                        elif self.reward_stop_time - self.reward_start_time >= 5:
                            self.missed_counter += 1
                            self.write_to_file("missed_trial", time)

                    elif status == "distractor_on":
                        self.distractor_on = True
                        self.rat_left_during_distractor = False
                        self.distractor_start_time = time_in_seconds(time)
                        self.write_to_file("distractor_time_started", time)
                    elif status == "distractor_off":
                        self.distractor_on = False
                        self.distractor_stop_time = time_in_seconds(time)
                        if not self.rat_left_during_distractor and self.distractor_stop_time - self.distractor_start_time >= 5:
                            self.correct_distractor_counter += 1
                            self.write_to_file("distractor_avoided", time)

    def write_to_file_memory(self, type_event, seconds):
        csv_name = str(self.report_path) + '/rat' + str(self.report_rank) + '_' + str(self.episode) + '.csv'
        if not os.path.exists(csv_name):
            with open(csv_name, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['time_stamp', 'event', 'missed', 'correct'])
        with open(csv_name, 'a') as fd:
            writer = csv.writer(fd)
            seconds = int(seconds)
            h = str(seconds // 3600)
            m = str((seconds % 3600) // 60)
            s = str((seconds % 3600) % 60)
            writer.writerow([str(self.episode) + "_" + h + ":" + m + ":" + s,
                             type_event, self.missed_counter, self.correct_counter])

    def process_command_memory(self, obs):
        if self.report_path is None:
            return
        instr = obs['INSTR']
        if instr:
            instr = json.loads(instr)
            for command_idx in range(1, instr['nCommands'] + 1):
                command = instr['Command' + str(command_idx)]['Command']

                if command == "Position":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = int(instr['Command' + str(command_idx)]['Opt']['Num2'])
                    self.position = instr['Command' + str(command_idx)]['Opt']['String1']

                    self.write_to_file(self.position, time)

                elif command == "Pickup":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = int(instr['Command' + str(command_idx)]['Opt']['Num2'])
                    name = instr['Command' + str(command_idx)]['Opt']['String1']
                    self.correct_counter += 1
                    self.write_to_file("Picked up " + name, time)

                elif command == "DoorStatus":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = int(instr['Command' + str(command_idx)]['Opt']['Num2'])
                    name = instr['Command' + str(command_idx)]['Opt']['String1']
                    status = instr['Command' + str(command_idx)]['Opt']['String2']
                    self.write_to_file(name + ' ' + status, time)

                elif command == "SetReward":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = int(instr['Command' + str(command_idx)]['Opt']['Num2'])
                    name = instr['Command' + str(command_idx)]['Opt']['String1']
                    self.write_to_file("Set reward " + name, time)

                elif command == "LostReward":
                    self.episode = int(instr['Command' + str(command_idx)]['Opt']['Num1'])
                    time = int(instr['Command' + str(command_idx)]['Opt']['Num2'])
                    name = instr['Command' + str(command_idx)]['Opt']['String1']
                    self.write_to_file("Lost reward " + name, time)

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
            infos = {'sound_status': self.sound_on,
                     'distractor_status': self.distractor_on,
                     'episode': {'r': self.total_reward,
                                 'l': self.len,
                                 't': time.time() - self.start}}
            self.reset()
            return self._last_observation, 0.0, False, infos
        obs = self._lab.observations()
        self._last_observation = obs[self._colors] if obs[self._colors] is not None else self._last_observation
        reward = self._lab.step(ACTION_LIST[action], num_steps=1)
        done = self.done(obs)
        self.process_command(obs)
        self.len += 1
        self.total_reward += reward
        if done:
            infos = {'sound_status': self.sound_on,
                     'distractor_status': self.distractor_on,
                     'episode': {'r': self.total_reward,
                                 'l': self.len,
                                 't': time.time() - self.start}}
            self.total_reward = 0.0
            self.len = 0
            self.start = time.time()
            self.episode += 1
        else:
            infos = {'sound_status': self.sound_on,
                     'distractor_status': self.distractor_on,}
        return self._last_observation, reward, done, infos

    def reset(self):
        self._lab.reset()
        if not self._lab.is_running():
            self._lab.reset()
        self._lab.step(ACTION_LIST[0], num_steps=1)
        self._last_observation = self._lab.observations()[self._colors]
        self.start = time.time()
        self.total_reward = 0.0
        self.len = 0

        self.sound_on = None
        self.distractor_on = None
        self.rat_left_base_during_reward_time = False
        self.rat_left_during_distractor = False
        self.episode = 0
        self.position = "base1"
        self.missed_counter = 0
        self.early_counter = 0
        self.late_counter = 0
        self.correct_counter = 0
        self.distractor_counter = 0
        self.correct_distractor_counter = 0
        return self._last_observation

    def seed(self, seed=None):
        self._lab.reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)

    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._lab.observations()[self._colors]
        # elif mode is 'human':
        #   pop up a window and render
        else:
            super(DeepmindLabEnv, self).render(mode=mode)  # just raise an exception

    def get_action_meanings(self):
        return english_names_of_actions


def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = [
    _action(0, 0, 0, 0, 0, 0, 0),  # noop
    _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
    _action(20, 0, 0, 0, 0, 0, 0),  # look_right
    # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
    # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
    # _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    # _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(0, 0, 0, 1, 0, 0, 0),  # forward
    # _action(  0,   0,  0, -1, 0, 0, 0), # backward
    # _action(  0,   0,  0,  0, 1, 0, 0), # fire
    # _action(  0,   0,  0,  0, 0, 1, 0), # jump
    # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
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
