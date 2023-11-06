from dataclasses import dataclass, field
from nostalgia.dynamine.data.generate import MultivariateSineCorrelations

@dataclass
class DynaMineConfig:

    data: MultivariateSineCorrelations = field(default_factory=MultivariateSineCorrelations)
    learning_rate: float = 1e-4
    loss:str = 'mine_orig'
    alpha:float = 0.01
    iters:int = 100

    #model
    time_embedding_dim:int = 19
    hidden_dim:int = 200
