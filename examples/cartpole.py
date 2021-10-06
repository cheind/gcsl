from typing import List

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
        self.goal = None
        self.goal_geom = None
        self.goal_transform = None
        self.update_goal = False

    def set_goal(self, goal: gcsl.Goal):
        self.goal = goal
        self.update_goal = True

    def render(self, mode="human", **kwargs):
        from gym.envs.classic_control import rendering

        if self.goal_geom is None:
            self.env.render(mode="rgb_array", **kwargs)  # prepare
            self.goal_geom = rendering.Line((0, 100 - 10), (0, 100 + 10))
            self.goal_geom.set_color(1.0, 0, 0)
            self.goal_transform = rendering.Transform()
            self.goal_geom.add_attr(self.goal_transform)
            self.env.viewer.add_geom(self.goal_geom)
        if self.update_goal:
            scale = 600 / (2 * 2.4)
            goalx = self.goal[0] * scale + 600 / 2.0
            self.update_goal = False
            self.goal_transform.set_translation(goalx, 0)

        return self.env.render(mode=mode, **kwargs)


class CartpolePolicyNet(torch.nn.Module):
    """The cartpole policy network outputting action-logits for state-goal inputs.
    Realized by a fully connected network with two hidden layers.
    """

    def __init__(self):
        super().__init__()
        self.logits = gcsl.make_fc_layers(
            6, [20, "A", "D", 100, "A", "D", 2], dropout=0.2
        )

    def forward(self, s, g, h):
        del h
        x = torch.cat((s, g), -1)
        logits = self.logits(x)
        return logits


def sample_goal():
    pos = np.random.uniform(-1.0, 1.0)
    pole_angle = 0.0
    return np.array([pos, pole_angle], dtype=np.float32)


def sample_goal_eval():
    pos = -1.0 if np.random.rand() < 0.5 else 1.0
    pole_angle = 0.0
    return np.array([pos, pole_angle], dtype=np.float32)


def relabel_goal(t0: gcsl.SAGHTuple, t1: gcsl.SAGHTuple) -> gcsl.Goal:
    s, _, _, _ = t1
    pos = s[0]
    pole_angle = s[2]
    return np.array([pos, pole_angle], dtype=np.float32)


def filter_trajectories(trajectories: List[gcsl.Trajectory]):
    ft = [t for t in trajectories if abs(t[-1][0][2]) < 0.1 and len(t) > 20]
    return ft


def main():
    # Create env
    env = gym.make("CartPole-v1")
    env = CartpoleGoalRenderWrapper(env)

    # Setup the policy-net
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=1e-3)

    # Create a policy-fn invoking our net. This will be called
    # during data collection and evaluation, but not used
    # in training.
    policy_fn = gcsl.make_policy_fn(net, egreedy=0.00)

    # Create a buffer for experiences
    buffer = gcsl.ExperienceBuffer(int(1e6))

    # Collected and store experiences from a random policy
    trajectories = gcsl.collect_trajectories(
        env=env,
        goal_sample_fn=sample_goal,
        policy_fn=lambda s, g, h: env.action_space.sample(),
        num_episodes=10,
        max_steps=50,
    )
    buffer.insert(trajectories)

    # Main GCSL loop
    pbar = trange(1, 10000, unit="steps")
    avg_rewards = 0.0
    avg_elen = 0.0
    for e in pbar:

        # np.random.seed(123)
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
                    policy_fn=policy_fn,
                    num_episodes=100,
                    max_steps=50,
                )
                net.train()
            buffer.insert(filter_trajectories(trajectories))

        if e % 1000 == 0:
            net.eval()
            avg_rewards, avg_elen = gcsl.evaluate_policy(
                env,
                goal_sample_fn=sample_goal_eval,
                policy_fn=policy_fn,
                num_episodes=20,
                max_steps=500,
                render_freq=20,
            )
            net.train()
        if e % 100 == 0:
            pbar.set_postfix(
                loss=loss.item(),
                avg_rew=f"{avg_rewards:.2f}",
                avg_elen=f"{avg_elen:.2f}",
            )
    env.close()


if __name__ == "__main__":
    main()
