import torch
import numpy as np
from typing import List
from dataclasses import dataclass,field,fields
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Example of usage
@dataclass
class MultivariateSineCorrelations:
    sample_size:int = 1000
    training_proportion:float = 0.8
    batch_size:int = 128

    dt:float = 1.0
    jitter:float = 1e-6
    number_of_timesteps:int = 100
    max_number_of_time_steps:int = 10

    p:List[float] = field(default_factory=lambda :[30.0,60.0,80.])
    l:List[float] = field(default_factory=lambda :[1.0,1.0,1.0])
    sigma_0:List[float] = field(default_factory=lambda :[.5,.5,.5])

    @property
    def dimension(self) -> int:
        return len(self.p)

def get_multivariate_timeseries(config:MultivariateSineCorrelations):
    timeseries = []
    rhos = []
    for dimension_index in range(config.dimension):
        p = config.p[dimension_index]
        l = config.l[dimension_index]
        sigma_0 = config.sigma_0[dimension_index]

        beta_sine_fix = lambda t: beta_sine(t, p, l, sigma_0)
        timeseries_batch = generate_timeseries_batch(config.sample_size,
                                                     config.number_of_timesteps,
                                                     beta_sine_fix).unsqueeze(-1)

        t_vals = torch.arange(config.number_of_timesteps) * config.dt
        rho_values = beta_sine(t_vals, p, l, 1.).unsqueeze(-1)

        timeseries.append(timeseries_batch)
        rhos.append(rho_values)
    timeseries = torch.cat(timeseries,axis=-1)
    rhos = torch.cat(rhos,axis=-1)
    return timeseries,rhos

class MultivariateTimeSeriesDataloader:

    def __init__(self,config:MultivariateSineCorrelations):
        self.config = config
        self.timeseries,self.rhos = get_multivariate_timeseries(config)
        # print(f'self.timeseries type: {self.timeseries.dtype}')
        timeseries_dataset = TensorDataset(self.timeseries)
        self.dataloader = DataLoader(timeseries_dataset,
                                     batch_size=config.batch_size)
        self.number_of_timesteps = config.number_of_timesteps
        self.dimension = config.dimension
        self.obtain_exact_mi()

    def obtain_exact_mi(self):
        self.exact_mi = torch.zeros(self.number_of_timesteps)
        for dimension_index in range(self.dimension):
            self.exact_mi += -.5*np.log(1.-(self.rhos[:,dimension_index])**2)
        return self.exact_mi

    def train(self):
        return self.dataloader

    def test(self):
        return None

def generate_timeseries_batch(batch_size, length, beta_fn, dt=1.0, jitter=1e-6):
    """
    Generate a batch of time series of given length with correlation function beta_fn.

    batch_size: int, number of time series to generate.
    length: int, length of each time series.
    beta_fn: Callable, the correlation function.
    dt: float, time step.
    jitter: float, a small value added to the diagonal for numerical stability.

    Returns: A batch of correlated time series of shape [batch_size, length].
    """
    # Create the covariance matrix using beta_fn
    t_vals = torch.arange(length) * dt
    Sigma = torch.zeros((length, length))
    for i in range(length):
        for j in range(length):
            Sigma[i, j] = beta_fn(torch.abs(t_vals[i] - t_vals[j]))

    # Add jitter to the diagonal for numerical stability
    Sigma += torch.eye(length) * jitter

    # Cholesky decomposition
    L = torch.linalg.cholesky(Sigma, upper=False)

    # Generate a batch of vectors of independent standard Gaussian random numbers
    z = torch.randn(batch_size, length)

    # Transform to get the batch of correlated sequences
    x = torch.mm(z, L.T)  # note the transpose on L

    return x


# Example of usage
def beta_example(t):
    """Example correlation function."""
    return torch.exp(-t / 10.0) ** 2


def beta_sine(t, p=30.0, l=1.0, sigma_0=1.):
    """
    Squared exponential with sinusoidal component kernel.

    t: tensor, time difference.
    p: float, period of the sinusoidal component.
    l: float, length scale.

    Returns: Correlation value.
    """
    return (sigma_0 ** 2) * torch.exp(-2 * torch.sin(torch.pi * t / p) ** 2 / l ** 2)

def shuffle_along_time(tensor):
    """
    Shuffle a batch of time series along the time direction.

    tensor: Tensor of shape [batch_size, length].

    Returns: A tensor of the same shape with each time series shuffled along the time direction.
    """
    shuffled_tensor = tensor.clone()
    for i in range(tensor.size(0)):
        shuffled_tensor[i] = tensor[i][torch.randperm(tensor.size(1))]
    return shuffled_tensor


if __name__=="__main__":
    config = MultivariateSineCorrelations()
    time_series = get_multivariate_timeseries(config)
