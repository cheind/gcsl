from typing import Deque
import torch
from typing import List, Tuple
from collections import deque




class ReplayBuffer:
    def __init__(self, max_elements: int = None) -> None:
        self.data = deque(maxlen=max_elements)

    def add(self, trajectories: List[List[Tuple]]):
        for traj in trajectories:
            for idx, 
        pass


class X:
    pass