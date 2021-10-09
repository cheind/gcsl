"""This module contains code to train/eval GCSL on a modified cartpole environment.

The goal of the modified cartpole environment ist to maintain an upright pole and
reach a specific cart location. A goal, g, is described by a tuple
    g = (pos,pole_angle_rad)
In training, we sample goal cart locations uniformly across the range [-1.5, 1.5). The 
target pole angle is always zero.

Training in this environment is tricky, since the ends episodes when the pole angle is
slightly off from zero. This prevents the agent from gaining experiences far from its
starting location that is always around the center of the world.
"""


from functools import partial
from pathlib import Path
from typing import List, Tuple
import itertools

import gym
import ray
import imageio
import numpy as np
import torch
import torch.nn
import torch.optim as O
from tqdm import trange

import gcsl


class CartpoleGoalRenderWrapper(gym.Wrapper):
    """A goal-enriched wrapper for the classic Cartpole environment.

    Adds support for rendering goal positions as well as a metric
    that computes the distances between state and goal. Note, the metric
    is only used for evaluation and plays no role in learning.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.goal_geom = None
        self.goal_transform = None
        self.env._max_episode_steps = (
            1500  # otherwise we might be finished at 200/500 steps.
        )

    def goal_metric(self, state: gcsl.State, goal: gcsl.Goal) -> float:
        """Measures the absolute deviation of state to goal."""
        return abs(state[0] - goal[0])

    def render(self, mode="human", **kwargs):
        """Modifies orginal cartpole env to support goal rendering"""
        from gym.envs.classic_control import rendering

        goal = kwargs.pop("goal", None)
        if goal is not None:
            if self.goal_geom is None:
                self.env.render(mode="rgb_array", **kwargs)  # prepare
                l, r, t, b = -5, 5, 5, -5
                self.goal_geom = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)]
                )
                self.goal_geom.set_color(1.0, 0, 0)
                self.goal_transform = rendering.Transform()
                self.goal_geom.add_attr(self.goal_transform)
                self.env.viewer.add_geom(self.goal_geom)
            scale = 600 / (2 * 2.4)
            goalx = goal[0] * scale + 600 / 2.0
            self.goal_transform.set_translation(goalx, 100)

        return self.env.render(mode=mode, **kwargs)


class CartpolePolicyNet(torch.nn.Module):
    """The cartpole policy network predicting action-logits for
    state-goal inputs.

    The architecture is a three small hidden layer FC network.
    """

    def __init__(self):
        super().__init__()
        self.logits = gcsl.make_fc_layers(
            6, [64, "A", 128, "A", "D", 256, "A", 2], dropout=0.1
        )

    def forward(self, s, g, h):
        del h
        x = torch.cat((s, g), -1)
        logits = self.logits(x)
        return logits


def sample_goal(xrange: Tuple[float, float] = (-1.5, 1.5)) -> gcsl.Goal:
    """Sample a new goal. In the cartpole environment a goal is composed of a
    cart-position and a pole angle."""
    pos = np.random.uniform(*xrange)
    pole_angle = 0.0
    return np.array([pos, pole_angle], dtype=np.float32)


def relabel_goal(t0: gcsl.SAGHTuple, t1: gcsl.SAGHTuple) -> gcsl.Goal:
    """Relabel the goal for `t0` using goal extracted from `t1`."""
    s, _, _, _ = t1
    pos = s[0]
    pole_angle = s[2]
    return np.array([pos, pole_angle], dtype=np.float32)


def filter_trajectories(trajectories: List[gcsl.Trajectory]):
    """Filter trajectories according to length. In our case we simply
    prefer longer sequences over shorter ones.

    Note, being verify picky about which trajectories enter the replay
    buffer may lead to faster learning, but a) requires domain knowledge
    and b) may result in less novel experiences, leading to overfitting of the
    policy on the few experiences in the buffer.

    In a sense, being selective here is like shaping the reward in RL.
    """
    ft = [t for t in trajectories if np.random.rand() > 10 / len(t)]
    return ft


@ray.remote
class RolloutHelper:
    def __init__(self):
        env = gym.make("CartPole-v1")
        self.env = CartpoleGoalRenderWrapper(env)
        self.net = CartpolePolicyNet()
        self.policy_fn = gcsl.make_policy_fn(self.net, greedy=False, tscaling=0.1)
        self.goal_sample_fn = sample_goal

    def collect_trajectories(
        self, state_dict: dict, num_episodes: int, max_steps: int
    ) -> List[gcsl.Trajectory]:
        self.net.load_state_dict(state_dict)
        with torch.no_grad():
            self.net.eval()
            trajs = gcsl.collect_trajectories(
                self.env,
                self.goal_sample_fn,
                self.policy_fn,
                num_episodes=num_episodes,
                max_steps=max_steps,
            )
            trajs = filter_trajectories(trajs)
            self.net.train()
            return trajs


def train_agent(args):
    ray.init()
    rollout_envs = [
        RolloutHelper.remote() for _ in range(int(ray.available_resources()["CPU"]))
    ]

    """Main training routine."""
    # Create env
    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env)

    # Setup the policy-net
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=args.lr)

    # Create a policy-fn invoking our net. This will be called
    # during data collection and evaluation, but not used in training.
    eval_policy_fn = gcsl.make_policy_fn(net, greedy=True)

    # Create a buffer for experiences
    buffer = gcsl.ExperienceBuffer(args.buffer_size)

    # Collected and store experiences from a random policy
    trajectories = gcsl.collect_trajectories(
        env=env,
        goal_sample_fn=partial(sample_goal, xrange=(-1.5, 1.5)),
        policy_fn=lambda s, g, h: env.action_space.sample(),
        num_episodes=50,
        max_steps=args.max_eps_steps,
    )
    buffer.insert(trajectories)

    # Main GCSL loop
    pbar = trange(1, args.num_gcsl_steps + 1, unit="steps")
    postfix_dict = {"agm": 0.0, "alen": 0.0, "neweps": 0, "loss": 0.0}
    pending_episode_ids = []
    for e in pbar:
        # Perform a single GCSL training step
        loss = gcsl.gcsl_step(
            net,
            opt,
            buffer,
            relabel_goal,
            batch_size=args.batch_size,
            max_relabel_horizon=args.max_relabel_horizon,
        )
        postfix_dict["loss"] = loss.item()

        # Update any pending results from rollout workers
        ready, not_ready = ray.wait(pending_episode_ids, timeout=0.01)
        if len(ready) > 0:
            ready_trajs = list(itertools.chain(*ray.get(ready)))
            buffer.insert(ready_trajs)
            postfix_dict["neweps"] = len(ready_trajs)
        pending_episode_ids = not_ready
        postfix_dict["pending"] = len(pending_episode_ids)

        if e % args.collect_freq == 0:
            # Every now and then, sample new experiences
            # First, store current model waits (blocking)
            state_id = ray.put(net.state_dict())
            # Distribute work across rollout helpers (not blocking)
            pending_episode_ids += [
                re.collect_trajectories.remote(
                    state_id,
                    args.num_eps_collect // len(rollout_envs),
                    args.max_eps_steps,
                )
                for re in rollout_envs
            ]
            del state_id

        if e % args.eval_freq == 0:
            # Evaluate the policy and save model
            net.eval()
            agm, alen = gcsl.evaluate_policy(
                env,
                goal_sample_fn=sample_goal,
                policy_fn=eval_policy_fn,
                num_episodes=args.num_eps_eval,
                max_steps=args.max_eps_steps,
                render_freq=args.render_freq,  # shows only last
            )
            postfix_dict["alen"] = alen
            postfix_dict["agm"] = agm
            net.train()
            torch.save(net.state_dict(), f"./tmp/cartpolenet_{e:05d}.pth")
        if e % 100 == 0:
            pbar.set_postfix(postfix_dict)
    env.close()


def eval_agent(args):
    """Main routine for rendering results."""

    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env)
    net = CartpolePolicyNet()
    net.load_state_dict(torch.load(args.weights))
    eval_policy_fn = gcsl.make_policy_fn(net, greedy=True)
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.dynamic_goal:
        # In case the goal is dynamic, we linearly interpolate the goal
        # position between xmin, xmax over max-steps
        goal_sample_fn = lambda: np.array([args.goal_xmin, 0.0], dtype=np.float32)

        def goal_dyn_fn(g, tdata):
            t, tmax = tdata
            pos = args.goal_xmin + (t / tmax) * (args.goal_xmax - args.goal_xmin)
            g[0] = pos
            return g

    else:
        goal_sample_fn = partial(sample_goal, xrange=(args.goal_xmin, args.goal_xmax))
        goal_dyn_fn = None

    result = gcsl.evaluate_policy(
        env,
        goal_sample_fn=goal_sample_fn,
        goal_dynamics_fn=goal_dyn_fn,
        policy_fn=eval_policy_fn,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render_freq=args.render_freq,
        return_images=args.save_gif,
    )
    env.close()
    avg_metric, avg_lens = result[:2]
    print("avg-metric", avg_metric, "avg-len", avg_lens)
    if args.save_gif:
        imageio.mimsave(f"./tmp/{Path(args.weights).stem}.gif", result[-1][::2], fps=60)


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser for training
    parser_train = subparsers.add_parser("train", help="train cartpole agent")
    parser_train.set_defaults(func=train_agent)
    parser_train.add_argument("-lr", type=float, default=5e-4, help="learning rate")
    parser_train.add_argument(
        "-num-gcsl-steps", type=int, default=int(1e5), help="number of GCSL steps"
    )
    parser_train.add_argument(
        "-max-eps-steps",
        type=int,
        default=500,
        help="maximum number of episode steps in eval and collection",
    )
    parser_train.add_argument(
        "-buffer-size", type=int, default=int(1e6), help="capacity of buffer"
    )
    parser_train.add_argument(
        "-collect-freq",
        type=int,
        default=100,
        help="collect new experiences every nth step",
    )
    parser_train.add_argument(
        "-num-eps-collect",
        type=int,
        default=100,
        help="number of episodes per evaluation step",
    )
    parser_train.add_argument(
        "-num-eps-eval",
        type=int,
        default=50,
        help="number of episodes per collection step",
    )
    parser_train.add_argument(
        "-eval-freq", type=int, default=5000, help="eval every nth step"
    )
    parser_train.add_argument(
        "-render-freq", type=int, default=50, help="render every nth episode of eval"
    )
    parser_train.add_argument(
        "-batch-size", type=int, default=512, help="batch-size per gcsl step"
    )
    parser_train.add_argument(
        "-max-relabel-horizon", type=int, default=None, help="maximum relabel horizon"
    )

    # Parser for evaluation
    parser_eval = subparsers.add_parser("eval", help="eval cartpole agent")
    parser_eval.set_defaults(func=eval_agent)
    parser_eval.add_argument("weights", type=Path, help="agent policy weights")
    parser_eval.add_argument(
        "-num-episodes", type=int, default=2, help="number of episodes to run"
    )
    parser_eval.add_argument(
        "-max-steps", type=int, default=500, help="max steps per episode"
    )
    parser_eval.add_argument(
        "-render-freq", type=int, default=1, help="render every nth episode"
    )
    parser_eval.add_argument("-goal-xmin", type=float, default=-1.5)
    parser_eval.add_argument("-goal-xmax", type=float, default=1.5)
    parser_eval.add_argument(
        "--save-gif", action="store_true", help="save animated gif"
    )
    parser_eval.add_argument("-seed", type=int, help="seed the rng.")
    parser_eval.add_argument(
        "--dynamic-goal",
        action="store_true",
        help="Slowly move the goal during episode between goal-xmin and goal-xmax.",
    )

    args = parser.parse_args()
    if args.command == "train":
        train_agent(args)
    else:
        eval_agent(args)


if __name__ == "__main__":
    main()
