#!/usr/bin/env python3

import torch
from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
from nostalgia.dynamine.estimators.dynamic_statistic_networks import DynamicStatisticsNetwork
from nostalgia.dynamine.estimators.dynamine import DynaMine
from nostalgia.dynamine.configs.dynamine_configs import DynaMineConfig
import csv
import os

# Define different configurations for sample_size and iters
iters = 1000
sample_sizes = [5096] #5096
dims = ["dim=20"]
batch_sizes = [128]
hidden_dim = 400
p = {"dim=3": [30.0,60.0,80.], "dim=20":[30.0,60.0,80.0,30.0,60.0,80.0,30.0,60.0,80.0,30.0,60.0,80.0,30.0,60.0,80.0,30.0,60.0,80.0,30.0,60.0]}
l = {"dim=3": [1.0,1.0,1.0], "dim=20":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]}
sigma_0 = {"dim=3": [.5,.5,.5], "dim=20":[.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,]}

# Run experiments for each configuration
for sample_size in sample_sizes:
    for dim in dims:
        for batch_size in batch_sizes:
            # Set the configuration for the current experiment
            config = DynaMineConfig()
            config.data.sample_size = sample_size
            config.data.p = p[dim]
            config.data.l = l[dim]
            config.data.sigma_0 = sigma_0[dim]
            config.iters = iters
            config.data.batch_size = batch_size
            config.hidden_dim = hidden_dim

            # Initialize the components with the current configuration
            dataloader = MultivariateTimeSeriesDataloader(config.data)
            dyna_statistic_network = DynamicStatisticsNetwork(config)
            dynamine = DynaMine(T=dyna_statistic_network, config=config, verbose=False)

            # Run the optimization and get the mutual information
            mi = dynamine.optimize(dataloader)
