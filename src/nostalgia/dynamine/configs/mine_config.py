from dataclasses import dataclass
from mutual_information.data.generate import MultivariateSineCorrelations

@dataclass
class MineConfig:

    data: MultivariateSineCorrelations = MultivariateSineCorrelations()
    learning_rate: float = 1e-4
    loss:str = 'mine_orig'
    alpha:float = 0.01
    iters:int = 100