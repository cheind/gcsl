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

import gym
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

    def __init__(self, env: gym.Env, max_steps: int = 1500) -> None:
        super().__init__(env)
        self.goal_geom = None
        self.goal_transform = None
        self.env._max_episode_steps = (
            max_steps  # otherwise we might be finished at 200/500 steps.
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

    The architecture is a simple two hidden layer FC network.
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


def sample_goal(xrange: Tuple[float, float] = (-1.5, 1.5)) -> gcsl.Goal:
    """Sample a new goal. In the cartpole environment a goal is composed of a
    cart-position and a pole angle."""
    pos = np.random.uniform(*xrange)
    pole_angle = 0.0
    return np.array([pos, pole_angle], dtype=np.float32)


def relabel_goal(t0: gcsl.SAGHTuple, t1: gcsl.SAGHTuple) -> gcsl.Goal:
    """Relabel the goal for `t0` from state of `t1`."""
    s, _, _, _ = t1
    pos = s[0]
    pole_angle = s[2]
    return np.array([pos, pole_angle], dtype=np.float32)


def sample_goal_coop(xrange: Tuple[float, float] = (5.0, 5.0)) -> gcsl.Goal:
    """Sample a new goal. In the coop cartpole environment a goal is composed of a
    target cart-velocity and a pole angle."""
    cart_vel = np.random.uniform(*xrange)
    pole_angle = 0.0
    return np.array([cart_vel, pole_angle], dtype=np.float32)


def relabel_goal_coop(t0: gcsl.SAGHTuple, t1: gcsl.SAGHTuple) -> gcsl.Goal:
    """Relabel the goal for `t0` from state of `t1`."""
    s, _, _, _ = t1
    cart_vel = s[1]
    pole_angle = s[2]
    return np.array([cart_vel, pole_angle], dtype=np.float32)


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


def train_agent(args):
    """Main training routine."""
    # Create env
    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env)

    # Setup the policy-net
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=args.lr)

    # Create a policy-fn invoking our net. This will be called
    # during data collection and evaluation, but not used in training.
    collect_policy_fn = gcsl.make_policy_fn(net, greedy=False, tscaling=0.1)
    eval_policy_fn = gcsl.make_policy_fn(net, greedy=True)
    relabel_fn = relabel_goal_coop if args.coop else relabel_goal
    sample_goal_fn = sample_goal_coop if args.coop else sample_goal

    # Create a buffer for experiences
    buffer = gcsl.ExperienceBuffer(args.buffer_size)

    # Collected and store experiences from a random policy
    trajectories = gcsl.collect_trajectories(
        env=env,
        goal_sample_fn=sample_goal_fn,
        policy_fn=lambda s, g, h: env.action_space.sample(),
        num_episodes=50,
        max_steps=args.max_eps_steps,
    )
    buffer.insert(trajectories)

    # Main GCSL loop
    pbar = trange(1, args.num_gcsl_steps + 1, unit="steps")
    postfix_dict = {"agm": 0.0, "alen": 0.0, "neweps": 0, "loss": 0.0}
    for e in pbar:
        # Perform a single GCSL training step
        loss = gcsl.gcsl_step(net, opt, buffer, relabel_fn)

        if e % args.collect_freq == 0:
            # Every now and then, sample new experiences
            with torch.no_grad():
                net.eval()
                trajectories = gcsl.collect_trajectories(
                    env,
                    goal_sample_fn=sample_goal_fn,
                    policy_fn=collect_policy_fn,
                    num_episodes=args.num_eps_collect,
                    max_steps=args.max_eps_steps,
                )
                net.train()
            # Optionally filter the trajectories
            new_episodes = filter_trajectories(trajectories)
            buffer.insert(new_episodes)
            postfix_dict["neweps"] = len(new_episodes)
            postfix_dict["loss"] = loss.item()
        if e % args.eval_freq == 0:
            # Evaluate the policy and save model
            net.eval()
            agm, alen = gcsl.evaluate_policy(
                env,
                goal_sample_fn=sample_goal_fn,
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

    elif args.coop:
        goal_sample_fn = partial(
            sample_goal_coop, xrange=(args.goal_xmin, args.goal_xmax)
        )
        goal_dyn_fn = None
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


def play(args):
    from pyglet.window import key as pygletkey

    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env, max_steps=100000000)
    net = CartpolePolicyNet()
    net.load_state_dict(torch.load(args.weights))
    policy_fn = gcsl.make_policy_fn(net, greedy=True)

    target_vel = 0.0
    target_vel_delta = 0.0
    exit_game = False

    def on_key_press(key, mod):
        nonlocal target_vel_delta
        if key == pygletkey.LEFT:
            target_vel_delta = -0.2
        elif key == pygletkey.RIGHT:
            target_vel_delta = 0.2

    def on_key_release(key, mod):
        nonlocal target_vel_delta, exit_game
        if key == pygletkey.LEFT:
            target_vel_delta = 0.0
        elif key == pygletkey.RIGHT:
            target_vel_delta = 0.0
        elif key == pygletkey.ESCAPE:
            exit_game = True

    def make_goal() -> gcsl.Goal:
        cart_vel = target_vel
        pole_angle = 0.0
        return np.array([cart_vel, pole_angle], dtype=np.float32)

    env.render("human")
    env.env.viewer.window.on_key_press = on_key_press
    env.env.viewer.window.on_key_release = on_key_release

    state = env.reset()
    imgs = []
    while True:
        target_vel += target_vel_delta
        goal = make_goal()
        action = policy_fn(state, goal, 0)
        state, _, done, _ = env.step(action)
        env.render(mode="human", goal=goal)
        if args.save_gif:
            imgs.append(env.render(mode="rgb_array", goal=goal))
        if done:
            env.reset()
            target_vel = 0.0
        if exit_game:
            break
    if args.save_gif:
        imageio.mimsave(f"./tmp/{Path(args.weights).stem}.gif", imgs, fps=30)
    env.close()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser for training
    parser_train = subparsers.add_parser("train", help="train cartpole agent")
    parser_train.set_defaults(func=train_agent)
    parser_train.add_argument("-lr", type=float, default=1e-3, help="learning rate")
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
        "--coop", action="store_true", help="train for the human/agent coop scenario"
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
    parser_eval.add_argument(
        "--coop", action="store_true", help="train for the human/agent coop scenario"
    )

    parser_play = subparsers.add_parser("play", help="play coop cartpole agent")
    parser_play.add_argument("weights", type=Path, help="agent policy weights")
    parser_play.add_argument(
        "--save-gif", action="store_true", help="save animated gif"
    )

    args = parser.parse_args()
    if args.command == "train":
        train_agent(args)
    elif args.command == "eval":
        eval_agent(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
