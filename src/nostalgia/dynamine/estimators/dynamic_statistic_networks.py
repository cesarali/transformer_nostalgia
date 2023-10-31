import torch
from torch import nn
from mutual_information.configs.dynamine_configs import DynaMineConfig
from mutual_information.models.temporal_networks.embedding_utils import transformer_timestep_embedding

class DynamicStatisticsNetwork(nn.Module):
    """
    """
    def __init__(self, config: DynaMineConfig):
        super().__init__()
        x_dim = config.data.dimension
        y_dim = config.data.dimension
        self.time_embedding_dim = config.time_embedding_dim
        self.statistics_network = nn.Sequential(
            nn.Linear(x_dim + y_dim + self.time_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, out_features=1)
        )
    def forward(self,X,Y,time,dim=1):
        time_emb = transformer_timestep_embedding(time,self.time_embedding_dim)

        XY = torch.cat([X,Y,time_emb], dim=1)
        T = self.statistics_network(XY)
        return T


if __name__=="__main__":
    from mutual_information.data.generate import MultivariateTimeSeriesDataloader
    config = DynaMineConfig()
    config.data.sample_size = 2000
    dataloader = MultivariateTimeSeriesDataloader(config.data)

    databatch = next(dataloader.train().__iter__())
    timeseries_batch = databatch[0]
    batch_size = timeseries_batch.size(0)
    random_time_indexes = torch.randint(0, config.data.max_number_of_time_steps, (batch_size,))
    X = timeseries_batch[:, 0, :]
    Y = timeseries_batch[range(batch_size), random_time_indexes, :]

    dyna_statistic_network = DynamicStatisticsNetwork(config)
    forward_ = dyna_statistic_network(X,Y,random_time_indexes)

    print(forward_)
