from typing import Callable, Deque
import torch
import gym
import numpy as np
from typing import List, Tuple, Any
from collections import deque
import time
import copy


def play(
    env: gym.Env,
    goal_sample_fn: Callable,
    policy_fn: Callable,
    num_episodes: int,
    max_steps: int = 50,
) -> List[List[Tuple]]:
    """Plays and returns trajectories sampled according to policy."""
    trajectories = []
    for _ in range(num_episodes):
        state = env.reset()
        goal = goal_sample_fn()
        traj = []
        for s in range(max_steps):
            action = policy_fn(state, goal)
            new_state, _, done, _ = env.step(action)
            if done:
                break
            else:
                traj.append(copy.copy((state, action)))
            state = new_state
        trajectories.append(traj)
    return trajectories


def eval_policy(
    env: gym.Env,
    goal_sample_fn: Callable,
    policy_fn: Callable,
    num_episodes: int,
    max_steps: int = 50,
    render: bool = False,
):
    for _ in range(num_episodes):
        goal = goal_sample_fn()
        state = env.reset()
        print(goal)
        for _ in range(max_steps):
            action = policy_fn(state, goal)
            state, _, done, _ = env.step(action)
            if render:
                env.render(mode="human")
                time.sleep(0.5 if done else 0.05)
            if done:
                break


class ReplayBuffer:
    def __init__(self, max_elements: int = None) -> None:
        self.data = deque(maxlen=max_elements)

    def add(self, trajectories: List[List[Tuple]]):
        """Add trajectories to the buffer."""
        for traj in trajectories:
            T = len(traj)
            for t, (state, action) in enumerate(traj):
                # We insert each element as tuple (o,a,h), where
                # h is the horizon. Even as old elements get dropped
                # from deque, elements h steps in the future will
                # remain valid.
                self.data.append((state, action, T - t - 1))

    def sample_and_relabel(
        self, batch_size: int, max_horizon: int = None, as_tensor: bool = True
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """Samples and relables (s,g,a,h) tuples from previous trajectories."""
        states, goals, actions, horizons = [], [], [], []
        indices = np.random.choice(len(self.data), size=batch_size)
        if max_horizon is None:
            max_horizon = 100000
        for idx in indices:
            s, a, h = self.data[idx]
            h = min(h, max_horizon)
            states.append(s)
            actions.append(a)
            horizons.append(h)
            goals.append(self.data[idx + h][0])
        if as_tensor:
            states = torch.tensor(states).float()
            goals = torch.tensor(goals).float()
            actions = torch.tensor(actions).long()
            horizons = torch.tensor(horizons).int()
        return states, goals, actions, horizons
