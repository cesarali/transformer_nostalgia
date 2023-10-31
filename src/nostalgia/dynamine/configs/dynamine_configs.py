from dataclasses import dataclass
from mutual_information.data.generate import MultivariateSineCorrelations

@dataclass
class DynaMineConfig:

    data: MultivariateSineCorrelations = MultivariateSineCorrelations()
    learning_rate: float = 1e-4
    loss:str = 'mine_orig'
    alpha:float = 0.01
    iters:int = 100

    #model
    time_embedding_dim:int = 19
    hidden_dim:int = 200
