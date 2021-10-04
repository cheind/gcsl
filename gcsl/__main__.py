from . import play, ReplayBuffer
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
        self.nn1 = torch.nn.Linear(5, 200, bias=False)  # obs + goal (position)
        self.nn2 = torch.nn.Linear(200, 2)
        self.bn1 = torch.nn.BatchNorm1d(200)

    def forward(self, s, g, h):
        del h
        x = torch.cat((s, g[:, :1]), -1)
        x = self.nn1(x)
        x = F.relu(self.bn1(x))
        x = self.nn2(x)
        return F.log_softmax(x, -1), D.Categorical(logits=x)

    def predict(self, s, g):
        s = torch.tensor(s).view(1, 4)
        g = torch.tensor(g).view(1, 1)
        _, d = self(s, g, None)
        return d.sample().item()


def main():
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(100000)
    trajectories = play(
        env,
        lambda: np.random.uniform(-4.8, 4.8),
        lambda s, g: env.action_space.sample(),
        num_episodes=5000,
    )
    buffer.add(trajectories)
    net = CartpolePolicyNet()
    opt = O.Adam(net.parameters(), lr=1e-3)

    for e in range(50000):
        s, g, a, h = buffer.sample_and_relabel(256)
        mask = h > 0

        logprobs, _ = net(s[mask], g[mask], h[mask])
        loss = F.nll_loss(logprobs, a[mask])
        loss.backward()
        opt.step()

        if e % 100 == 0:
            with torch.no_grad():
                net.eval()
                trajectories = play(
                    env,
                    lambda: np.random.uniform(-4.8, 4.8),
                    lambda s, g: net.predict(s, g),
                    num_episodes=100,
                )
                net.train()
            buffer.add(trajectories)

        print(loss.item())


if __name__ == "__main__":
    main()