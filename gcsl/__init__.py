import copy
import itertools
import time
from collections import deque
from typing import Any, Callable, List, Literal, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.distributions as D
import torch.optim
import torch.nn
import torch.nn.functional as F

State = Any
Goal = State
Action = int
Horizon = int
SAGHTuple = Tuple[State, Action, Goal, Horizon]
Trajectory = List[SAGHTuple]
PolicyFn = Callable[[State, Goal, Horizon], Action]
GoalSampleFn = Callable[[], Goal]
GoalUpdateFn = Callable[[Goal, Tuple[int, int]], Goal]
GoalRelabelFn = Callable[[SAGHTuple, SAGHTuple], Goal]


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
        for t in range(max_steps):
            action = policy_fn(state, goal, t)
            new_state, _, done, _ = env.step(action)
            traj.append(copy.copy((state, action, goal)))
            state = new_state
            if done:
                break
        # Add horizon info
        T = len(traj)
        horizons = T - np.arange(T) - 1
        traj = [t + (h,) for t, h in zip(traj, horizons)]
        # Finalize trajectory
        trajectories.append(traj)
    return trajectories


@torch.no_grad()
def evaluate_policy(
    env: gym.Env,
    goal_sample_fn: GoalSampleFn,
    policy_fn: PolicyFn,
    num_episodes: int,
    max_steps: int = 50,
    render_freq: bool = False,
    return_images: bool = False,
    goal_dynamics_fn: GoalUpdateFn = None,
) -> Union[Tuple[float, float], Tuple[float, float, List[np.ndarray]]]:
    """Evaluate the policy in the given environment.
    Returns average final goal metric and average episode lengths."""

    if goal_dynamics_fn is None:
        goal_dynamics_fn = lambda g, _: g

    goal_metrics = []
    all_lengths = []
    all_images = []
    for e in range(1, num_episodes + 1):
        goal = goal_sample_fn()
        render = e % render_freq == 0
        state = env.reset()
        for t in range(max_steps):
            goal = goal_dynamics_fn(goal, (t, max_steps))
            action = policy_fn(state, goal, t)
            state, _, done, _ = env.step(action)
            if render:
                if return_images:
                    img = env.render(mode="rgb_array", goal=goal)
                    all_images.append(img)
                else:
                    env.render(mode="human", goal=goal)
                    time.sleep(0.5 if done else 0.01)
            if done:
                print(state[1])
                break
        if hasattr(env, "goal_metric"):
            goal_metrics.append(env.goal_metric(state, goal))
        all_lengths.append(t + 1)
    if return_images:
        return np.mean(goal_metrics), np.mean(all_lengths), all_images
    else:
        return np.mean(goal_metrics), np.mean(all_lengths)


class ExperienceBuffer:
    """Experience buffer of limited capacity.

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

    def insert(self, trajectories: List[Trajectory]):
        """Append experiences. This might remove oldest values
        if capacity of the buffer is reached."""
        for t in itertools.chain(*trajectories):
            self.memory.append(t)

    def sample(
        self,
        num_experiences: int,
        goal_relabel_fn: GoalRelabelFn,
        max_horizon: int = None,
    ) -> List[SAGHTuple]:
        """Uniform randomly sample N (state,action,goal,horizon) tuples."""
        indices = np.random.choice(len(self.memory), size=num_experiences)
        if max_horizon is None:
            max_horizon = np.iinfo(int).max
        tuples = [self._relabel(idx, goal_relabel_fn, max_horizon) for idx in indices]
        return tuples

    def __len__(self):
        return len(self.memory)

    def _relabel(
        self, idx: int, goal_relabel_fn: GoalRelabelFn, max_horizon: int
    ) -> SAGHTuple:
        t0 = self.memory[idx]
        s, a, _, h = t0
        if h > 0:
            # If not last element of trajectory, we can sample
            # a new horizon, which defines the target tuple.
            h = int(np.random.randint(1, min(h + 1, max_horizon)))
        t1 = self.memory[idx + h]
        # Note, h(t0) >= h(t1)
        g = goal_relabel_fn(t0, t1)
        return (s, a, g, h)


def sample_buffers(
    buffers: Union[ExperienceBuffer, Sequence[ExperienceBuffer]],
    num_experiences: int,
    goal_relabel_fn: GoalRelabelFn,
    max_horizon: int = None,
    buf_probs: Sequence[float] = None,
) -> List[SAGHTuple]:
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
        b.sample(n, goal_relabel_fn, max_horizon=max_horizon)
        for b, n in zip(buffers, num_samples_per_buffer)
    ]
    return list(itertools.chain(*nested_tuples))


def to_tensor(
    tuples: List[SAGHTuple],
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


def gcsl_step(
    net: torch.nn.Module,
    opt: torch.optim.Optimizer,
    buffers: Union[ExperienceBuffer, Sequence[ExperienceBuffer]],
    relabel_goal_fn: GoalRelabelFn,
) -> torch.Tensor:
    """Performs a single training step in the GCSL regime."""
    s, a, g, h = to_tensor(sample_buffers(buffers, 512, relabel_goal_fn))
    mask = h > 0  # Only consider samples which are not final states

    opt.zero_grad()
    logits = net(s[mask], g[mask], h[mask])
    loss = F.cross_entropy(logits, a[mask])
    loss.backward()
    opt.step()
    return loss


def make_fc_layers(
    infeatures: int,
    arch: List[Union[int, Literal["D", "A", "N"]]],
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
        elif d == "N":
            layers.append(torch.nn.BatchNorm1d(last_c))
        else:
            layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*layers)


def make_policy_fn(
    net: torch.nn.Module, greedy: bool = False, tscaling: float = 0.1
) -> PolicyFn:
    """Creates a default policy function wrapping a torch.nn.Module returning action-logits.
    This method will be called for a single (s,a,h) tuple and its components may not be
    torch types.
    """

    def predict(s: State, g: Goal, h: Horizon):
        s = torch.tensor(s).unsqueeze(0)
        g = torch.tensor(g).unsqueeze(0)
        logits = net(s, g, h)
        if greedy:
            return torch.argmax(logits).item()
        else:
            scaled_logits = logits * (1 - tscaling)
            return D.Categorical(logits=scaled_logits).sample().item()

    return predict
