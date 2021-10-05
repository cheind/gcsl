from torch.nn.modules.linear import Linear
from . import play, eval_policy, ReplayBuffer
import gym
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as O


class CartpolePolicyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Sequential(
            torch.nn.Linear(6, 400, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(400, 300, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(300, 2, bias=True),
        )

    def forward(self, s, g, h):
        del h
        x = torch.cat((s, g[..., 2:]), -1)
        logits = self.logits(x)
        return logits

    def predict(self, s, g, egreedy: float = 0.0):
        s = torch.tensor(s).view(1, 4)
        g = torch.tensor(g).view(1, 4)
        logits = self(s, g, None)
        if np.random.rand() > egreedy:
            return torch.argmax(logits).item()
        else:
            return D.Categorical(logits=logits).sample().item()


def sample_cartpole_goal():
    return np.array([np.random.uniform(-0.5, 0.5), 0.0, 0.0, 0.0], dtype=np.float32)


def main():
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(1000000)
    trajectories = play(
        env,
        sample_cartpole_goal,
        lambda s, g: env.action_space.sample(),
        num_episodes=10,
    )
    buffer.add(trajectories)
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=1e-4)

    for e in range(50000):
        s, g, a, h = buffer.sample_and_relabel(256)
        mask = h > 0

        logits = net(s[mask], g[mask], h[mask])
        loss = F.cross_entropy(logits, a[mask])
        loss.backward()
        opt.step()

        if e % 100 == 0:
            with torch.no_grad():
                net.eval()
                trajectories = play(
                    env,
                    sample_cartpole_goal,
                    lambda s, g: net.predict(s, g, egreedy=0.02),
                    num_episodes=1,
                )
                net.train()
            buffer.add(trajectories)
        if e % 1000 == 0 and loss < 0.3:
            with torch.no_grad():
                net.eval()
                eval_policy(
                    env,
                    sample_cartpole_goal,
                    lambda s, g: net.predict(s, g),
                    num_episodes=3,
                    max_steps=500,
                    render=True,
                )
                net.train()

        print(loss.item())


if __name__ == "__main__":
    main()