import gcsl
import gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn
import torch.nn.functional as F
import torch.optim as O
from tqdm import trange


class CartpolePolicyNet(torch.nn.Module):
    """The cartpole policy network.
    A simple fully connected network with two hidden layers.
    """

    def __init__(self):
        super().__init__()
        self.logits = gcsl.make_fc_layers(5, [40, "A", "D", 50, "A", "D", 2])

    def forward(self, s, g, h):
        del h
        x = torch.cat((s, g[..., 2:3]), -1)
        logits = self.logits(x)
        return logits

    def predict(self, s, g, egreedy: float = 0.0) -> int:
        s = torch.tensor(s).view(1, 4)
        g = torch.tensor(g).view(1, 4)
        logits = self(s, g, None)
        if np.random.rand() > egreedy:
            return torch.argmax(logits).item()
        else:
            return D.Categorical(logits=logits).sample().item()


def sample_cartpole_goal():
    return np.array([np.random.uniform(-0.5, 0.5), 0.0, 0.0, 0.0], dtype=np.float32)


def filter_trajectories(trajectories):
    return [t for t in trajectories if abs(t[-1][0][2]) < 0.1]


def main():
    # Create env
    env = gym.make("CartPole-v1")

    # Setup the policy-net
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=1e-4)

    # Create a buffer for experiences
    buffer = gcsl.ExperienceBuffer(int(1e6))

    # Collected and store experiences from a random policy
    trajectories = gcsl.collect_trajectories(
        env=env,
        goal_sample_fn=sample_cartpole_goal,
        policy_fn=lambda s, g: env.action_space.sample(),
        num_episodes=100,
        max_steps=100,
    )
    buffer.insert(trajectories)

    # Main GCSL loop
    pbar = trange(1, 10000, unit="steps")
    avg_rewards = 0.0
    avg_elen = 0.0
    for e in pbar:

        # Sample a batch
        s, a, g, h = gcsl.to_tensor(gcsl.sample_buffers(buffer, 256))
        mask = h > 0

        # Perform stochastic gradient step
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
                    goal_sample_fn=sample_cartpole_goal,
                    policy_fn=lambda s, g: net.predict(s, g, egreedy=0.0),
                    num_episodes=10,
                    max_steps=100,
                )
                net.train()
            buffer.insert(filter_trajectories(trajectories))

        if e % 1000 == 0:
            net.eval()
            avg_rewards, avg_elen = gcsl.evaluate_policy(
                env,
                sample_cartpole_goal,
                lambda s, g: net.predict(s, g),
                num_episodes=20,
                max_steps=200,
                render_freq=20,
            )
            net.train()
        pbar.set_postfix(
            loss=loss.item(),
            avg_rew=f"{avg_rewards:.2f}",
            avg_elen=f"{avg_elen:.2f}",
        )
    env.close()


if __name__ == "__main__":
    main()
