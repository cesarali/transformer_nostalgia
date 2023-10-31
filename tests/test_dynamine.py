import pytest
import torch
from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
from nostalgia.dynamine.estimators.dynamic_statistic_networks import DynamicStatisticsNetwork
from nostalgia.dynamine.estimators.dynamine import DynaMine
from nostalgia.dynamine.configs.dynamine_configs import DynaMineConfig

class TestDynaMine:

    def test_init_dynamine(self):
        config = DynaMineConfig()
        config.data.sample_size = 2000
        config.iters = 5

        dataloader = MultivariateTimeSeriesDataloader(config.data)

        dyna_statistic_network = DynamicStatisticsNetwork(config)
        dynamine = DynaMine(T=dyna_statistic_network, config=config)
        mi = dynamine.optimize(dataloader)





