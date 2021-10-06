from typing import List
from functools import partial

from numpy.core.fromnumeric import var

import gcsl
import gym
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as O
from tqdm import trange


class CartpoleGoalRenderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.goal_geom = None
        self.goal_transform = None

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
    """The cartpole policy network outputting action-logits for state-goal inputs.
    Realized by a fully connected network with two hidden layers.
    """

    def __init__(self):
        super().__init__()
        self.logits = gcsl.make_fc_layers(
            6, [300, "A", "D", 400, "A", "D", 2], dropout=0.2
        )

    def forward(self, s, g, h):
        del h
        x = torch.cat((s, g), -1)
        logits = self.logits(x)
        return logits


def sample_goal(xrange=(-0.5, 0.5)):
    pos = np.random.uniform(*xrange)
    pole_angle = 0.0
    return np.array([pos, pole_angle], dtype=np.float32)


def relabel_goal(t0: gcsl.SAGHTuple, t1: gcsl.SAGHTuple) -> gcsl.Goal:
    s, _, _, _ = t1
    pos = s[0]
    pole_angle = s[2]
    return np.array([pos, pole_angle], dtype=np.float32)


def filter_trajectories(trajectories: List[gcsl.Trajectory]):
    ft = [t for t in trajectories if np.random.rand() > 10 / len(t)]
    return ft
    # ft = [t for t in trajectories if abs(t[-1][0][2]) < 0.1]
    # return ft


def train_agent(args):
    # Create env
    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env)

    # Setup the policy-net
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=1e-4)

    # Create a policy-fn invoking our net. This will be called
    # during data collection and evaluation, but not used
    # in training.
    collect_policy_fn = gcsl.make_policy_fn(net, greedy=False, tscaling=0.1)
    eval_policy_fn = gcsl.make_policy_fn(net, greedy=True)

    # Create a buffer for experiences
    buffer = gcsl.ExperienceBuffer(int(1e6))

    # Collected and store experiences from a random policy
    trajectories = gcsl.collect_trajectories(
        env=env,
        goal_sample_fn=partial(sample_goal, xrange=(-1.5, 1.5)),
        policy_fn=lambda s, g, h: env.action_space.sample(),
        num_episodes=50,
        max_steps=400,
    )
    buffer.insert(trajectories)

    # Main GCSL loop
    pbar = trange(1, 100000, unit="steps")
    postfix_dict = {"agm": 0.0, "alen": 0.0, "neweps": 0, "loss": 0.0}
    for e in pbar:

        # Sample a batch
        s, a, g, h = gcsl.to_tensor(gcsl.sample_buffers(buffer, 512, relabel_goal))
        mask = h > 0

        # Perform stochastic gradient step
        opt.zero_grad()
        logits = net(s[mask], g[mask], h[mask])
        loss = F.cross_entropy(logits, a[mask])
        loss.backward()
        opt.step()

        if e % 100 == 0:
            # Every now and then, sample new trajectories
            # Add trajectories according to some filter criterium
            with torch.no_grad():
                net.eval()
                trajectories = gcsl.collect_trajectories(
                    env,
                    goal_sample_fn=sample_goal,
                    policy_fn=collect_policy_fn,
                    num_episodes=100,
                    max_steps=400,
                )
                net.train()
            new_episodes = filter_trajectories(trajectories)
            buffer.insert(new_episodes)
            postfix_dict["neweps"] = len(new_episodes)
            postfix_dict["loss"] = loss.item()
        if e % 500 == 0:
            net.eval()
            agm, alen = gcsl.evaluate_policy(
                env,
                goal_sample_fn=sample_goal,
                policy_fn=eval_policy_fn,
                num_episodes=20,
                max_steps=500,
                render_freq=20,
            )
            postfix_dict["alen"] = alen
            postfix_dict["agm"] = agm
            net.train()
            torch.save(net.state_dict(), f"./tmp/cartpolenet_{e:05d}.pth")
        if e % 100 == 0:
            pbar.set_postfix(postfix_dict)
    env.close()


def eval_agent(args):

    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env)
    net = CartpolePolicyNet()
    net.load_state_dict(torch.load(args.weights))
    eval_policy_fn = gcsl.make_policy_fn(net, greedy=True)

    agm, alen = gcsl.evaluate_policy(
        env,
        goal_sample_fn=partial(sample_goal, xrange=(args.goal_xmin, args.goal_xmax)),
        policy_fn=eval_policy_fn,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render_freq=args.render_freq,
    )
    env.close()


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="train cartpole agent")
    parser_train.set_defaults(func=train_agent)

    parser_eval = subparsers.add_parser("eval", help="eval cartpole agent")
    parser_eval.add_argument("weights", type=Path, help="agent policy weights")
    parser_eval.add_argument(
        "-num-episodes", type=int, default=20, help="number of episodes to run"
    )
    parser_eval.add_argument(
        "-max-steps", type=int, default=500, help="max steps per episode"
    )
    parser_eval.add_argument(
        "-render-freq", type=int, default=1, help="render every nth episode"
    )
    parser_eval.add_argument("-goal-xmin", type=float, default=-0.5)
    parser_eval.add_argument("-goal-xmax", type=float, default=0.5)
    parser_eval.set_defaults(func=eval_agent)
    args = parser.parse_args()
    if args.command == "train":
        train_agent(args)
    else:
        eval_agent(args)


if __name__ == "__main__":
    main()
