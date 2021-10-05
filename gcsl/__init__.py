import copy
import itertools
import multiprocessing as mp
import time
from collections import deque
from typing import Any, Callable, List, Literal, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn

State = Any
Goal = State
Action = Any
StateActionTuple = Tuple[State, Action]
StateActionGoalHorizonTuple = Tuple[State, Action, State, int]
Trajectory = List[StateActionTuple]
PolicyFn = Callable[[State, Goal], int]
GoalSampleFn = Callable[[], Goal]


def collect_trajectories(
    env: gym.Env,
    goal_sample_fn: GoalSampleFn,
    policy_fn: PolicyFn,
    num_episodes: int,
    max_steps: int = 50,
) -> List[List[Tuple]]:
    """Collect trajectories by interacting with the environment."""
    trajectories = []
    for _ in range(num_episodes):
        state = env.reset()
        goal = goal_sample_fn()
        traj = []
        for s in range(max_steps):
            action = policy_fn(state, goal)
            new_state, _, done, _ = env.step(action)
            traj.append(copy.copy((state, action)))
            state = new_state
            if done:
                break
        trajectories.append(traj)
    return trajectories


@torch.no_grad()
def evaluate_policy(
    env: gym.Env,
    goal_sample_fn: Callable,
    policy_fn: PolicyFn,
    num_episodes: int,
    max_steps: int = 50,
    render_freq: bool = False,
) -> Tuple[float, float]:
    """Evaluate the policy in the given environment.
    Returns average rewards and episode lengths."""
    all_rewards = []
    all_lengths = []
    for e in range(1, num_episodes + 1):
        goal = goal_sample_fn()
        render = e % render_freq == 0
        state = env.reset()
        rewards = []
        if render:
            print(goal)
        for t in range(max_steps):
            action = policy_fn(state, goal)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if render:
                env.render(mode="human")
                time.sleep(0.5 if done else 0.01)
            if done:
                break
        all_lengths.append(t + 1)
        all_rewards.append(np.sum(rewards))
    return np.mean(all_rewards), np.mean(all_lengths)


class ExperienceBuffer:
    """Simple experience buffer of potentially limited capacity.

    This buffers stores (state, action, horizon) tuples of trajectories
    in a flat memory layout. Here horizon is time (offset >= 0) to tuple
    representing the final trajectory state.

    Once the capacity is reached oldest tuples will get discarded first,
    leaving potentially partial trajectories in memory. Since we only
    required that a future state is available for relabeling, this does
    not pose a problem.
    """

    def __init__(self, max_experiences) -> None:
        self.memory = deque(maxlen=max_experiences)
        self.lock = mp.Lock()

    def insert(self, trajectories: List[Trajectory]):
        tuples = []
        for traj in trajectories:
            T = len(traj)
            # We insert each element as tuple (o,a,h), where
            # h is the horizon. Even as old elements get dropped
            # from deque, elements h steps in the future will
            # remain valid.
            horizons = T - np.arange(len(traj)) - 1
            tuples += [(s, a, h) for (s, a), h in zip(traj, horizons)]
        with self.lock:
            for t in tuples:
                self.memory.append(t)

    def sample(
        self, num_experiences: int, max_horizon: int = None
    ) -> List[StateActionGoalHorizonTuple]:
        """Uniform randomly sample N (state,action,goal,horizon) tuples."""
        with self.lock:
            indices = np.random.choice(len(self.memory), size=num_experiences)
            if max_horizon is None:
                max_horizon = 100000
            tuples = [self._sample(idx, max_horizon) for idx in indices]
        return tuples

    def __len__(self):
        return len(self.memory)

    def _sample(self, idx, max_horizon: int) -> StateActionGoalHorizonTuple:
        s, a, h = self.memory[idx]
        if h > 0:
            # If not last element of trajectory
            h = int(np.random.randint(1, min(h + 1, max_horizon)))
        g = self.memory[idx + h][0]
        return (s, a, g, h)


def sample_buffers(
    buffers: Union[ExperienceBuffer, Sequence[ExperienceBuffer]],
    num_experiences: int,
    max_horizon: int = None,
    buf_probs: Sequence[float] = None,
) -> List[StateActionGoalHorizonTuple]:
    """Sample experiences from a number of buffers.

    This function is particularily useful if multiple buffers are to be sampled
    from. In this case, per default, the expected number of experiences sampled
    from each buffers is proportional to the buffer length.
    """

    if isinstance(buffers, ExperienceBuffer):
        buffers = [buffers]
    if buf_probs is None:
        buf_probs = np.array([len(b) for b in buffers]).astype(float)
        buf_probs /= buf_probs.sum()
    else:
        buf_probs = np.ndarray(buf_probs)
    num_samples_per_buffer = np.random.multinomial(num_experiences, buf_probs)
    nested_tuples = [
        b.sample(n, max_horizon=max_horizon)
        for b, n in zip(buffers, num_samples_per_buffer)
    ]
    return list(itertools.chain(*nested_tuples))


def to_tensor(
    tuples: List[StateActionGoalHorizonTuple],
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.IntTensor]:
    """Converts lists of (state,action,goal,horizon) tuples to separate tensors."""
    if len(tuples) > 0:
        states, actions, goals, horizons = zip(*tuples)
    else:
        states, actions, goals, horizons = [], [], [], []
    states = torch.tensor(states)
    goals = torch.tensor(goals)
    actions = torch.tensor(actions).long()
    horizons = torch.tensor(horizons).int()
    return states, actions, goals, horizons


def make_fc_layers(
    infeatures: int,
    arch: List[Union[int, Literal["D", "A"]]],
    dropout: float = 0.1,
    activation: Type[torch.nn.Module] = torch.nn.ReLU,
) -> torch.nn.Sequential:
    """Helper function to create a fully connected network from an architecture description."""
    layers = []
    last_c = infeatures
    for d in arch:
        if isinstance(d, int):
            layers.append(torch.nn.Linear(last_c, d))
            last_c = d
        elif d == "A":
            layers.append(activation())
        else:
            layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*layers)
