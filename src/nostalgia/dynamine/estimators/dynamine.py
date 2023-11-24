import os
import math
import torch
import numpy as np
import csv
import datetime

from torch import nn
from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
from nostalgia.dynamine.configs.dynamine_configs import DynaMineConfig
from nostalgia.dynamine.estimators.ema import ema_loss
from .utils import get_device, print_time

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DynaMine(nn.Module):

    def __init__(self, T, config:DynaMineConfig, method=None, verbose=False, print_interval=100):
        super().__init__()
        self.device = get_device()
        self.to(self.device)
        self.running_mean = 0
        self.loss = config.loss
        self.alpha = config.alpha
        self.method = 'concat'
        self.iters = config.iters
        self.lr = config.learning_rate
        self.max_number_of_time_steps = config.data.max_number_of_time_steps
        self.T = T
        self.verbose = verbose
        self.print_interval = print_interval
        self.start_time = None

    def forward(self, x, z, z_marg=None,time=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]
        # print(f'time shape: {time.shape}. t: {time.tolist()}')
        t = self.T(x, z,time).mean()
        # print(f't shape: {t.shape}. t: {t.item()}')
        t_marg = self.T(x, z_marg,time)

        if self.loss in ['mine_orig']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None,time=None):
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg,time)
        return mi

    def optimize(self, dataloader:MultivariateTimeSeriesDataloader, opt=None):
        mi_timeseries = []

        # Define the CSV filename
        csv_filename = "scripts/evaluations/performance_dynamine/experiment_log.csv"

        if self.verbose:
            print(f'Sample size: {dataloader.config.sample_size}')
            print(f'Dimension: {dataloader.config.dimension}')

       # Open the CSV file once and write headers if needed
        file_exists = os.path.isfile(csv_filename)
        csvfile = open(csv_filename, 'a', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=[
            'timestamp', 'sample_size', 'training_proportion', 'batch_size', 'dt', 'jitter',
            'number_of_timesteps', 'max_number_of_time_steps', 'p', 'l', 'sigma_0', 'dimension',
            'time_steps_ahead', 'exact_mi', 'iter', 'estimated_mi'])
        if not file_exists:
            writer.writeheader()

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        print_time()
        for iter in range(1, self.iters + 1):
            self.train()
            if iter%self.print_interval==0:
                print(f"Iter Index {iter}")
                print_time()
            mu_mi = 0
            for batch_idx, timeseries_batch in enumerate(dataloader.train()):
                timeseries_batch = timeseries_batch[0].to(self.device) #take tensor out of list
                batch_size = timeseries_batch.size(0)
                if self.verbose:
                    print(f'batch_idx: {batch_idx}')
                    print(f'shape of timeseries_batch with batch_idx {batch_idx}: {timeseries_batch.shape}')
                    # print(f'first batch, first example, first 10: \n{timeseries_batch[0][0][0:10]}')
                    # print(f'timeseries_batch: {timeseries_batch}')

                random_time_indexes = torch.randint(0, self.max_number_of_time_steps, (batch_size,)).to(self.device)
                if self.verbose:
                    print(f'random_time_indexes shape: {random_time_indexes.shape}')
                    print(f'random_time_indexes (5 examples): {random_time_indexes[0:5]}')
                x = timeseries_batch[:, 0, :].to(self.device)
                if self.verbose:
                    print(f'x shape: {x.shape}')
                    # print(f'x: {x}')
                y = timeseries_batch[range(batch_size), random_time_indexes, :].to(self.device)
                if self.verbose:
                    print(f'y shape: {y.shape}')
                    # print(f'y: {y}')

                opt.zero_grad()
                loss = self.forward(x, y, time=random_time_indexes)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()

            #=========================================
            # FULL TIME SERIES
            #=========================================
            self.eval()
            if hasattr(dataloader, "exact_mi"):
                for time_steps_ahead in range(self.max_number_of_time_steps):
                    X = dataloader.timeseries[:, 0, :].to(self.device)
                    Y = dataloader.timeseries[:, time_steps_ahead, :].to(self.device)
                    time = torch.full((X.size(0),), time_steps_ahead).to(self.device)
                    # print(f'time: {time}')
                    final_mi = self.mi(X, Y, time=time)
                    if self.verbose:
                            print(f'\nINPUT: \nX shape: {X.shape}. \nY shape: {Y.shape}.\ntime shape: {time.shape}.\n OUTPUT: \nfinal_mi shape: {final_mi}')
                    mi_timeseries.append(final_mi.item())

                    # Prepare the data to be logged
                    log_data = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'sample_size': dataloader.config.sample_size,
                        'training_proportion': dataloader.config.training_proportion,
                        'batch_size': dataloader.config.batch_size,
                        'dt': dataloader.config.dt,
                        'jitter': dataloader.config.jitter,
                        'number_of_timesteps': dataloader.config.number_of_timesteps,
                        'max_number_of_time_steps': dataloader.config.max_number_of_time_steps,
                        'p': str(dataloader.config.p),
                        'l': str(dataloader.config.l),
                        'sigma_0': str(dataloader.config.sigma_0),
                        'dimension': dataloader.config.dimension,
                        'time_steps_ahead': time_steps_ahead,
                        'exact_mi': str(dataloader.exact_mi[time_steps_ahead].item()),
                        'iter': iter,
                        'estimated_mi': final_mi.item()
                    }

                    if ((iter%self.print_interval==0) or (iter == self.iters)) and (time_steps_ahead%3==0):
                            print(f'time steps ahead: {time_steps_ahead}. Estimated MI: {final_mi.item():.2f}. True MI: {dataloader.exact_mi[time_steps_ahead].item():.2f}')

                    # Write the log data to the CSV file
                    try:
                        with open(csv_filename, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
                            writer.writerow(log_data)
                    except FileNotFoundError:
                        print(f"File not found: {csv_filename}")
                        print("Hint: Make sure to run this script from the root of the repository.")
        # print(mi_timeseries)
        # print(len(mi_timeseries))
        return mi_timeseries
