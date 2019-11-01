# Gym wrapper for DeepMind Lab environments
Gym implementation of connector to Deepmind lab

## Getting Started
Install the newest package by running the following command from the parent directory.
```
pip install -e gym-deepmindlab-env/
```

In your project:
```
import gym
import gym_deepmindlab

env = gym.make('DeepmindLabSeekavoidArena01-v0')

# Use the environment
observation = env.reset()
```

## Suppoted Environments
This is the list of supported environments:
- DeepmindLabSoundTaskZero-v0
- DeepmindLabLtChasm-v0
- DeepmindLabLtHallwaySlope-v0
- DeepmindLabLtHorseshoeColor-v0
- DeepmindLabLtSpaceBounceHard-v0
- DeepmindLabNavMazeRandomGoal01-v0
- DeepmindLabNavMazeRandomGoal02-v0
- DeepmindLabNavMazeRandomGoal03-v0
- DeepmindLabNavMazeStatic01-v0
- DeepmindLabNavMazeStatic02-v0
- DeepmindLabSeekavoidArena01-v0
- DeepmindLabStairwayToMelon-v0

You can add new environments to `LEVELS` in `gym_deepmindlab/__init__.py`, and create them by calling `DeepmindLab<YourEnv>-v0`. 
Note that `<YourEnv>` has to be CamelCased.

## Arguments
You can specify the screen size when creating a new environment:
```
import gym
import gym_deepmindlab

env = gym.make('DeepmindLabSeekavoidArena01-v0', width = 224, height = 224)

# Use the environment
observation = env.reset()
```

## Thanks
Thanks to https://github.com/deepmind/lab for such a great work.
