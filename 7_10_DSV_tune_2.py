import json
import numpy as np
import os
import torch
import tqdm
import math
import numpy as np
import gpytorch
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import random


from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy,MeanFieldVariationalDistribution,CholeskyVariationalDistribution,LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.models.exact_gp import GP
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from matplotlib import pyplot as plt
import argparse

# current_device = torch.cuda.current_device()
# torch.cuda.set_device(current_device)

params = argparse.ArgumentParser()
params.add_argument('-num_search', type=int, default=300, help='iteration time of random searching ')
params.add_argument('-repeat_time', type=int, default=1, help='repeat time of random searching ')
params.add_argument('-cross_split', type=int, default=2, help='number of cross validation split ')
params.add_argument('-params_epoch', type=int, default=10, help=' epoch number of finding parameters ')
params.add_argument('-optimal_epoch', type=int, default=10, help=' epoch number of optimal parameters ')

args = params.parse_args()

num_search = args.num_search
repeat_time = args.repeat_time
cross_split = args.cross_split
params_epoch = args.params_epoch
optimal_epoch = args.optimal_epoch


torch.manual_seed(8)
smoke_test = ('CI' in os.environ)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_file = 'april.json'
with open(data_file) as f:
    data = json.load(f)
collect_data = {}
receiver_position = torch.empty(0, dtype=torch.float)
# print(receiver_position)
RSS_value = torch.empty(0)
# print(RSS_value)
for key, value in data.items():
    rx_data = value['rx_data']
    # print(rx_data)
    metadata = value['metadata']
    # print(metadata)
    power = metadata[0]['power']
    # print(power)
    base_station = rx_data[2][3]
    # print(base_station)
    tr_cords = torch.tensor(value["tx_coords"]).float()
    # print(tr_cords)
    if base_station == 'cbrssdr1-bes-comp' and power == 1: # 'cbrssdr1-bes-comp', 'cbrssdr1-honors-comp'
        RSS_sample = torch.tensor([rx_data[2][0]]).float()
        # print(RSS_sample)
        RSS_value = torch.cat((RSS_value, RSS_sample), dim=0)
        # print(RSS)
        # print(RSS.shape)
        receiver_position = torch.cat((receiver_position, tr_cords), dim=0)
        # print(location)
        # print(location.shape)
# print(RSS_value)
# print(RSS_value.shape) # torch.Size([326])
# print(receiver_position)
# print(receiver_position.shape) # torch.Size([326, 2])
# process RSS_value and receiver_position data
RSS_value = RSS_value.view(RSS_value.size(0),1)
for i in range(len(receiver_position)):
    receiver_position[i][0] = (receiver_position[i][0] - 40.75) * 1000
    receiver_position[i][1] = (receiver_position[i][1] + 111.83) * 1000
# print('original RSS value',RSS_value)
# print(RSS_value.shape)

shuffle_index = torch.randperm(len(receiver_position))
receiver_position = receiver_position[shuffle_index].to(device)
RSS_value = RSS_value[shuffle_index].to(device)

train_x = receiver_position[30:,:] # cbrssdr1-honors-comp: 409, 359, 309, 189
train_y = RSS_value[30:,:]
test_x = receiver_position[0:30,:]
test_y = RSS_value[0:30,:]

# normalize the train x and test x
mean_norm_x, std_norm_x = train_x.mean(),train_x.std()
train_x = (train_x - mean_norm_x) / (std_norm_x)
test_x = (test_x - mean_norm_x) / (std_norm_x)

# normalize the train y and test y
mean_norm, std_norm = train_y.mean(),train_y.std()
train_y = (train_y - mean_norm) / (std_norm)
test_y = (test_y - mean_norm) / (std_norm)

train_y = 10 ** (train_y / 20)*100000
test_y = 10 ** (test_y / 20)*100000

mean_norm_decimal, std_norm_decimal = train_y.mean(),train_y.std()
train_y = (train_y - mean_norm_decimal) / (std_norm_decimal)
test_y = (test_y - mean_norm_decimal) / (std_norm_decimal)

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

print(RSS_value.shape) # 326, 485
# print('decimal train x',train_x)
# print('decimal train y',train_y)


def initialize_inducing_inputs(X, M):
    kmeans = KMeans(n_clusters=M)
    # print('kmeans',kmeans)
    kmeans.fit(X.cpu())
    # print('kmeans.fit(X)', kmeans.fit(X))
    inducing_inputs = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    return inducing_inputs
def _init_pca(X, latent_dim):
    U, S, V = torch.pca_lowrank(X, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(X, V[:,:latent_dim]))

#Deep Gaussian Process
class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing =160, linear_mean = False): # 40, 70,100, 160;
        # inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        if input_dims == train_x.shape[-1]:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)
            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(train_x, num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
        else:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)
            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(_init_pca(train_x, input_dims).detach(), num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)

        batch_shape = torch.Size([output_dims])
        # mean_field variational distribution
        # variational_distribution = MeanFieldVariationalDistribution(
        #     num_inducing_points=num_inducing,
        #     batch_shape=batch_shape
        # )
        # variational_distribution = MeanFieldVariationalDistribution.initialize_variational_distribution()

        # print('variational variational_mean',variational_distribution.variational_mean)
        # print('variational variational_stddev', variational_distribution.variational_stddev)
        # print(variational_distribution.covariance_matrix)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        # include the inducing points
        mean_x =self.mean_module(x)
        # print('mean_x',mean_x)
        covar_x = self.covar_module(x)
        # print('covar_x',covar_x)
        return MultivariateNormal(mean_x, covar_x)


num_tasks = train_y.size(-1)
num_hidden_dgp_dims = 1

class MultitaskDeepGP(DeepGP):
    def __init__(self,train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims = train_x_shape[-1],
            output_dims= num_hidden_dgp_dims,
            linear_mean=True
        )

        # second_hidden_layer = DGPHiddenLayer(
        #     input_dims=hidden_layer.output_dims,
        #     output_dims=num_hidden_dgp_dims+1,
        #     linear_mean=True
        # )
        #
        # third_hidden_layer = DGPHiddenLayer(
        #     input_dims=second_hidden_layer.output_dims+train_x_shape[-1],
        #     output_dims=num_hidden_dgp_dims+2,
        #     linear_mean=True
        # )

        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=True
        )
        super().__init__()

        self.hidden_layer = hidden_layer
        # self.second_hidden_layer = second_hidden_layer
        # self.third_hidden_layer = third_hidden_layer
        self.last_layer = last_layer

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs,**kwargs):

        hidden_rep1 = self.hidden_layer(inputs)
        # print('22',inputs.shape)
        # print('11',hidden_rep1)
        # hidden_rep2 = self.second_hidden_layer(hidden_rep1, **kwargs)
        # hidden_rep3 = self.third_hidden_layer(hidden_rep2, inputs, **kwargs)
        output = self.last_layer(hidden_rep1)
        return output
    def predict(self, test_x):
        with torch.no_grad():
            preds = model.likelihood(model(test_x)).to_data_independent_dist()
        return preds.mean.mean(0), preds.variance.mean(0)

model = MultitaskDeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42}')
for constraint_name, constraint in model.named_constraints():
    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')

# hiden_lengthscale_choice = [-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0]
# last_lengthscale_choice  = [-0.1, 0.1]
# hiden_outputscale_choise = [-3.0, -2.0, -1.0, -0.1]
# last_outputscale_choise  = [-1.0, -0.1]
# likelihood_noise_range   = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

hiden_lengthscale_choice = [-4.0, -6.0, -5.0] # want to try -5.5
hiden_outputscale_choise = [-3.0, -2.0, -1.0]
last_lengthscale_choice  = [-0.1, 0.1, 4.0]
last_outputscale_choise  = [1.0, -0.1, -1.0]
likelihood_noise_range   = [0.05, 0.1, 1.5]

# hiden_lengthscale_choice = [-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0]
# last_lengthscale_choice  = [-0.1, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# hiden_outputscale_choise = [-3.0, -2.0, -1.0, -0.1, 0.01, 0.1, 1.0, 2.0, 3.0]
# last_outputscale_choise  = [-1.0, -0.1, 0.01, 0.1, 1.0, 2.0, 3.0]
# likelihood_noise_range   = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# lengthscale=[0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
# outputscale=[0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
# likelihood_noise_range=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]

# best_loss = float('inf')
best_train = float('inf')
second_best_train = float('inf')
third_best_train = float('inf')
fourth_best_train = float('inf')
fifth_best_train = float('inf')
sixth_best_train = float('inf')
seventh_best_train = float('inf')
eighth_best_train = float('inf')
ninth_best_train = float('inf')
tenth_best_train = float('inf')

best_hyperparams = None
best_result = None
second_best_hyperparams = None
second_best_result = None
third_best_hyperparams = None
third_best_result = None
fourth_best_hyperparams = None
fourth_best_result = None
fifth_best_hyperparams = None
fifth_best_result = None
sixth_best_hyperparams = None
sixth_best_result = None
seventh_best_hyperparams = None
seventh_best_result = None
eighth_best_hyperparams = None
eighth_best_result = None
ninth_best_hyperparams = None
ninth_best_result = None
tenth_best_hyperparams = None
tenth_best_result = None


# cross_split = 4
kf = KFold(n_splits=cross_split, shuffle=True)

# # random searching and cross validation
# method = ['random searching and cross validation']
# # num_search = 100 # 50, 80，100
# grid_times = 0
# for search_time in range(num_search):
#     hidden_layer_lengthscale = random.choice(hiden_lengthscale_choice)
#     hidden_layer_outputscale = random.choice(hiden_outputscale_choise)
#     last_layer_lengthscale = random.choice(last_lengthscale_choice)
#     last_layer_outputscale = random.choice(last_outputscale_choise)
#     likelihood_noise_value = random.choice(likelihood_noise_range)
#     for time in range(repeat_time):
#         grid_times = grid_times + 1
#         total_train_rmse = 0.0
#         total_test_rmse = 0.0
#         total_loss = 0.0
#         print(grid_times,
#               '. hidden_lengthscale: ', hidden_layer_lengthscale,
#               '; hidden_outputscale: ', hidden_layer_outputscale,
#               '; last_lengthscale: ', last_layer_lengthscale,
#               '; last_outputscale: ', last_layer_outputscale,
#               '; noise: ', likelihood_noise_value)
#         for train_index, val_index in kf.split(train_x):
#             train_x_cross, test_x_cross = train_x[train_index], train_x[val_index]
#             train_y_cross, test_y_cross = train_y[train_index], train_y[val_index]
#             model = MultitaskDeepGP(train_x_cross.shape)
#             if torch.cuda.is_available():
#                 model = model.cuda()
#             hypers = {
#                 'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
#                     [[[hidden_layer_lengthscale, hidden_layer_lengthscale]]]).to(device),  # -3,
#                 'hidden_layer.covar_module.raw_outputscale': torch.tensor([hidden_layer_outputscale]).to(device),
#                 'last_layer.covar_module.raw_outputscale': torch.tensor([last_layer_outputscale]).to(device),
#                 # 0, 1, 0.5
#                 'last_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
#                     [[[last_layer_lengthscale]]]).to(device),
#                 'likelihood.raw_task_noises': torch.tensor([likelihood_noise_value]).to(device),
#                 'likelihood.raw_noise': torch.tensor([likelihood_noise_value]).to(device),
#             }
#             model.initialize(**hypers)
#             model.train()
#             optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#             mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y_cross.size(0))).to(device)
#             # num_epochs = 1 if smoke_test else 2000 best
#             epochs_iter = tqdm.tqdm(range(params_epoch), desc='Epoch')
#             loss_set = np.array([])
#
#             for i in epochs_iter:
#                 optimizer.zero_grad()
#                 output = model(train_x_cross)
#                 loss = -mll(output, train_y_cross)
#                 epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
#                 loss_set = np.append(loss_set, loss.item())
#                 loss.backward()
#                 optimizer.step()
#             total_loss += loss.item()
#             model.eval()
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 predictions, predictive_variance = model.predict(test_x_cross.float())
#             predictions = predictions * std_norm_decimal + mean_norm_decimal
#             test_y_cross = test_y_cross * std_norm_decimal + mean_norm_decimal
#             predictions = 20 * torch.log10(predictions / 100000)
#             test_y_cross = 20 * torch.log10(test_y_cross / 100000)
#
#             predictions = predictions * std_norm + mean_norm
#             test_y_cross = test_y_cross * std_norm + mean_norm
#             # print('noise_covar',noise_convar)
#             # print(predictive_variance)
#             for task in range(0, 1):
#                 # print(task)
#                 # print(predictions[:, task])
#
#                 test_rmse = torch.mean(torch.pow(predictions[:, task] - test_y_cross[:, task], 2)).sqrt()
#                 total_test_rmse += test_rmse.item()
#             model.eval()
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 predictions, predictive_variance = model.predict(train_x_cross.float())
#             predictions = predictions * std_norm_decimal + mean_norm_decimal
#             train_y_cross = train_y_cross * std_norm_decimal + mean_norm_decimal
#             predictions = 20 * torch.log10(predictions / 100000)
#             train_y_cross = 20 * torch.log10(train_y_cross / 100000)
#
#             predictions = predictions * std_norm + mean_norm
#             train_y_cross = train_y_cross * std_norm + mean_norm
#             for task in range(0, 1):
#                 # print(task)
#                 # print(predictions[:, task])
#                 train_rmse = torch.mean(torch.pow(predictions[:, task] - train_y_cross[:, task], 2)).sqrt()
#                 total_train_rmse += train_rmse.item()
#             # normalize the train x and test x
#             mean_norm_x, std_norm_x = train_x_cross.mean(), train_x_cross.std()
#             train_x_cross = (train_x_cross - mean_norm_x) / (std_norm_x)
#             test_x_cross = (test_x_cross - mean_norm_x) / (std_norm_x)
#
#             # normalize the train y and test y
#             mean_norm, std_norm = train_y_cross.mean(), train_y_cross.std()
#             train_y_cross = (train_y_cross - mean_norm) / (std_norm)
#             test_y_cross = (test_y_cross - mean_norm) / (std_norm)
#
#             train_y_cross = 10 ** (train_y_cross / 20) * 100000
#             test_y_cross = 10 ** (test_y_cross / 20) * 100000
#
#             mean_norm_decimal, std_norm_decimal = train_y_cross.mean(), train_y_cross.std()
#             train_y_cross = (train_y_cross - mean_norm_decimal) / (std_norm_decimal)
#             test_y_cross = (test_y_cross - mean_norm_decimal) / (std_norm_decimal)
#
#         print(grid_times,
#               '. hidden_lengthscale: ', hidden_layer_lengthscale,
#               '; hidden_outputscale: ', hidden_layer_outputscale,
#               '; last_lengthscale: ', last_layer_lengthscale,
#               '; last_outputscale: ', last_layer_outputscale,
#               '; noise: ', likelihood_noise_value)
#         print('    loss:', total_loss / cross_split,
#               ': test rmse: ', total_test_rmse / cross_split,
#               '; train rmse: ', total_train_rmse / cross_split)
#         if (total_train_rmse / cross_split) < best_train:
#             tenth_best_train = ninth_best_train
#             ninth_best_train = eighth_best_train
#             eighth_best_train = seventh_best_train
#             seventh_best_train = sixth_best_train
#             sixth_best_train = fifth_best_train
#             fifth_best_train = fourth_best_train
#             fourth_best_train = third_best_train
#             third_best_train = second_best_train
#             second_best_train = best_train
#             best_train = total_train_rmse / cross_split
#
#             tenth_best_hyperparams = ninth_best_hyperparams
#             ninth_best_hyperparams = eighth_best_hyperparams
#             eighth_best_hyperparams = seventh_best_hyperparams
#             seventh_best_hyperparams = sixth_best_hyperparams
#             sixth_best_hyperparams = fifth_best_hyperparams
#             fifth_best_hyperparams = fourth_best_hyperparams
#             fourth_best_hyperparams = third_best_hyperparams
#             third_best_hyperparams = second_best_hyperparams
#             second_best_hyperparams = best_hyperparams
#             best_hyperparams = {
#                 'hidden_layer_lengthscale': hidden_layer_lengthscale,
#                 'hidden_layer_outputscale': hidden_layer_outputscale,
#                 'last_layer_lengthscale': last_layer_lengthscale,
#                 'last_layer_outputscale': last_layer_outputscale,
#                 'likelihood_noise': likelihood_noise_value,
#             }
#
#             tenth_best_result = ninth_best_result
#             ninth_best_result = eighth_best_result
#             eighth_best_result = seventh_best_result
#             seventh_best_result = sixth_best_result
#             sixth_best_result = fifth_best_result
#             fifth_best_result = fourth_best_result
#             fourth_best_result = third_best_result
#             third_best_result = second_best_result
#             second_best_result = best_result
#             best_result = {
#                 'loss': total_loss / cross_split,
#                 'test rmse': total_test_rmse / cross_split,
#                 'train rmse': total_train_rmse / cross_split
#             }


# grid searching and cross validation
method = ['grid searching and cross validation']
grid_times = 0
for hidden_layer_lengthscale in hiden_lengthscale_choice:
    for hidden_layer_outputscale in hiden_outputscale_choise:
        for last_layer_lengthscale in last_lengthscale_choice:
            for last_layer_outputscale in last_outputscale_choise:
                for likelihood_noise_value in likelihood_noise_range:
                    for time in range(repeat_time):
                        grid_times = grid_times+1
                        total_train_rmse = 0.0
                        total_test_rmse = 0.0
                        total_loss = 0.0
                        print(grid_times,
                              '. hidden_lengthscale: ', hidden_layer_lengthscale,
                              '; hidden_outputscale: ', hidden_layer_outputscale,
                              '; last_lengthscale: ', last_layer_lengthscale,
                              '; last_outputscale: ', last_layer_outputscale,
                              '; noise: ', likelihood_noise_value)
                        for train_index, val_index in kf.split(train_x):
                            train_x_cross, test_x_cross = train_x[train_index], train_x[val_index]
                            train_y_cross, test_y_cross = train_y[train_index], train_y[val_index]
                            model = MultitaskDeepGP(train_x_cross.shape)
                            if torch.cuda.is_available():
                                model = model.cuda()
                            hypers = {
                                'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
                                    [[[hidden_layer_lengthscale, hidden_layer_lengthscale]]]).to(device),  # -3,
                                'hidden_layer.covar_module.raw_outputscale': torch.tensor([hidden_layer_outputscale]).to(device),
                                'last_layer.covar_module.raw_outputscale': torch.tensor([last_layer_outputscale]).to(device),
                                # 0, 1, 0.5
                                'last_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
                                    [[[last_layer_lengthscale]]]).to(device),
                                'likelihood.raw_task_noises': torch.tensor([likelihood_noise_value]).to(device),
                                'likelihood.raw_noise': torch.tensor([likelihood_noise_value]).to(device),
                            }
                            model.initialize(**hypers)
                            model.train()
                            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                            mll = DeepApproximateMLL(
                                VariationalELBO(model.likelihood, model, num_data=train_y_cross.size(0))).to(device)
                            # num_epochs = 1 if smoke_test else 100
                            # num_epochs = 1 if smoke_test else 2000 best
                            epochs_iter = tqdm.tqdm(range(params_epoch), desc='Epoch')
                            loss_set = np.array([])
                            for i in epochs_iter:
                                optimizer.zero_grad()
                                output = model(train_x_cross)
                                loss = -mll(output, train_y_cross)
                                epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
                                loss_set = np.append(loss_set, loss.item())
                                loss.backward()
                                optimizer.step()
                            total_loss += loss.item()
                            model.eval()
                            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                                predictions, predictive_variance = model.predict(test_x_cross.float())
                            predictions = predictions * std_norm_decimal + mean_norm_decimal
                            test_y_cross = test_y_cross * std_norm_decimal + mean_norm_decimal
                            predictions = 20 * torch.log10(predictions / 100000)
                            test_y_cross = 20 * torch.log10(test_y_cross / 100000)

                            predictions = predictions * std_norm + mean_norm
                            test_y_cross = test_y_cross * std_norm + mean_norm
                            # print('noise_covar',noise_convar)
                            # print(predictive_variance)
                            for task in range(0, 1):
                                # print(task)
                                # print(predictions[:, task])

                                test_rmse = torch.mean(
                                    torch.pow(predictions[:, task] - test_y_cross[:, task], 2)).sqrt()
                                total_test_rmse += test_rmse.item()
                            model.eval()
                            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                                predictions, predictive_variance = model.predict(train_x_cross.float())
                            predictions = predictions * std_norm_decimal + mean_norm_decimal
                            train_y_cross = train_y_cross * std_norm_decimal + mean_norm_decimal
                            predictions = 20 * torch.log10(predictions / 100000)
                            train_y_cross = 20 * torch.log10(train_y_cross / 100000)

                            predictions = predictions * std_norm + mean_norm
                            train_y_cross = train_y_cross * std_norm + mean_norm
                            for task in range(0, 1):
                                # print(task)
                                # print(predictions[:, task])
                                train_rmse = torch.mean(
                                    torch.pow(predictions[:, task] - train_y_cross[:, task], 2)).sqrt()
                                total_train_rmse += train_rmse.item()
                            # normalize the train x and test x
                            mean_norm_x, std_norm_x = train_x_cross.mean(), train_x_cross.std()
                            train_x_cross = (train_x_cross - mean_norm_x) / (std_norm_x)
                            test_x_cross = (test_x_cross - mean_norm_x) / (std_norm_x)

                            # normalize the train y and test y
                            mean_norm, std_norm = train_y_cross.mean(), train_y_cross.std()
                            train_y_cross = (train_y_cross - mean_norm) / (std_norm)
                            test_y_cross = (test_y_cross - mean_norm) / (std_norm)

                            train_y_cross = 10 ** (train_y_cross / 20) * 100000
                            test_y_cross = 10 ** (test_y_cross / 20) * 100000

                            mean_norm_decimal, std_norm_decimal = train_y_cross.mean(), train_y_cross.std()
                            train_y_cross = (train_y_cross - mean_norm_decimal) / (std_norm_decimal)
                            test_y_cross = (test_y_cross - mean_norm_decimal) / (std_norm_decimal)

                        print(grid_times,
                              '. hidden_lengthscale: ', hidden_layer_lengthscale,
                              '; hidden_outputscale: ', hidden_layer_outputscale,
                              '; last_lengthscale: ', last_layer_lengthscale,
                              '; last_outputscale: ', last_layer_outputscale,
                              '; noise: ', likelihood_noise_value)
                        print('    loss:', total_loss / cross_split,
                              ': test rmse: ', total_test_rmse / cross_split,
                              '; train rmse: ', total_train_rmse / cross_split)
                        if (total_train_rmse / cross_split) < best_train:
                            tenth_best_train = ninth_best_train
                            ninth_best_train = eighth_best_train
                            eighth_best_train = seventh_best_train
                            seventh_best_train = sixth_best_train
                            sixth_best_train = fifth_best_train
                            fifth_best_train = fourth_best_train
                            fourth_best_train = third_best_train
                            third_best_train = second_best_train
                            second_best_train = best_train
                            best_train = total_train_rmse / cross_split

                            tenth_best_hyperparams = ninth_best_hyperparams
                            ninth_best_hyperparams = eighth_best_hyperparams
                            eighth_best_hyperparams = seventh_best_hyperparams
                            seventh_best_hyperparams = sixth_best_hyperparams
                            sixth_best_hyperparams = fifth_best_hyperparams
                            fifth_best_hyperparams = fourth_best_hyperparams
                            fourth_best_hyperparams = third_best_hyperparams
                            third_best_hyperparams = second_best_hyperparams
                            second_best_hyperparams = best_hyperparams
                            best_hyperparams = {
                                'hidden_layer_lengthscale': hidden_layer_lengthscale,
                                'hidden_layer_outputscale': hidden_layer_outputscale,
                                'last_layer_lengthscale': last_layer_lengthscale,
                                'last_layer_outputscale': last_layer_outputscale,
                                'likelihood_noise': likelihood_noise_value,
                            }

                            tenth_best_result = ninth_best_result
                            ninth_best_result = eighth_best_result
                            eighth_best_result = seventh_best_result
                            seventh_best_result = sixth_best_result
                            sixth_best_result = fifth_best_result
                            fifth_best_result = fourth_best_result
                            fourth_best_result = third_best_result
                            third_best_result = second_best_result
                            second_best_result = best_result
                            best_result = {
                                'loss': total_loss / cross_split,
                                'test rmse': total_test_rmse / cross_split,
                                'train rmse': total_train_rmse / cross_split
                            }



# # random searching and separate validation
# method = ['random searching and separate validation']
# model = MultitaskDeepGP(train_x.shape)
# if torch.cuda.is_available():
#     model = model.cuda()
# # num_search = 10 # 50, 100
# grid_times = 0
# for search_time in range(num_search):
#     hidden_layer_lengthscale = random.choice(lengthscale)
#     hidden_layer_outputscale = random.choice(outputscale)
#     last_layer_lengthscale = random.choice(lengthscale)
#     last_layer_outputscale = random.choice(outputscale)
#     likelihood_noise_value = random.choice(likelihood_noise_range)
#     for time in range(repeat_time):
#         grid_times = grid_times + 1
#         hypers = {
#             'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
#                 [[[hidden_layer_lengthscale, hidden_layer_lengthscale]]]).to(device),  # -3,
#             'hidden_layer.covar_module.raw_outputscale': torch.tensor([hidden_layer_outputscale]).to(device),
#             'last_layer.covar_module.raw_outputscale': torch.tensor([last_layer_outputscale]).to(device),
#             # 0, 1, 0.5
#             'last_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
#                 [[[last_layer_lengthscale]]]).to(device),
#             'likelihood.raw_task_noises': torch.tensor([likelihood_noise_value]).to(device),
#             'likelihood.raw_noise': torch.tensor([likelihood_noise_value]).to(device),
#         }
#         model.initialize(**hypers)
#         model.train()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#         mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0))).to(device)
#         # num_epochs = 1 if smoke_test else 100
#         # num_epochs = 1 if smoke_test else 2000 best
#         epochs_iter = tqdm.tqdm(range(params_epoch), desc='Epoch')
#         loss_set = np.array([])
#         for i in epochs_iter:
#             optimizer.zero_grad()
#             output = model(train_x)
#             loss = -mll(output, train_y)
#             epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
#             loss_set = np.append(loss_set, loss.item())
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             predictions, predictive_variance = model.predict(test_x.float())
#         predictions = predictions * std_norm_decimal + mean_norm_decimal
#         test_y = test_y * std_norm_decimal + mean_norm_decimal
#         predictions = 20 * torch.log10(predictions / 100000)
#         test_y = 20 * torch.log10(test_y / 100000)
#
#         predictions = predictions * std_norm + mean_norm
#         test_y = test_y * std_norm + mean_norm
#         # print('noise_covar',noise_convar)
#         # print(predictive_variance)
#         for task in range(0, 1):
#             # print(task)
#             # print(predictions[:, task])
#
#             test_rmse = torch.mean(torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
#         model.eval()
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             predictions, predictive_variance = model.predict(train_x.float())
#         predictions = predictions * std_norm_decimal + mean_norm_decimal
#         train_y = train_y * std_norm_decimal + mean_norm_decimal
#         predictions = 20 * torch.log10(predictions / 100000)
#         train_y = 20 * torch.log10(train_y / 100000)
#
#         predictions = predictions * std_norm + mean_norm
#         train_y = train_y * std_norm + mean_norm
#         for task in range(0, 1):
#             # print(task)
#             # print(predictions[:, task])
#             train_rmse = torch.mean(torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
#         print(grid_times,
#               '. hidden_lengthscale: ', hidden_layer_lengthscale,
#               '; hidden_outputscale: ', hidden_layer_outputscale,
#               '; last_lengthscale: ', last_layer_lengthscale,
#               '; last_outputscale: ', last_layer_outputscale,
#               '; noise: ', likelihood_noise_value)
#         print('    loss:', loss.item(),
#               ': test rmse: ', test_rmse,
#               '; train rmse: ', train_rmse)
#         if train_rmse < best_train:
#             tenth_best_train = ninth_best_train
#             ninth_best_train = eighth_best_train
#             eighth_best_train = seventh_best_train
#             seventh_best_train = sixth_best_train
#             sixth_best_train = fifth_best_train
#             fifth_best_train = fourth_best_train
#             fourth_best_train = third_best_train
#             third_best_train = second_best_train
#             second_best_train = best_train
#             best_train = train_rmse
#
#             tenth_best_hyperparams = ninth_best_hyperparams
#             ninth_best_hyperparams = eighth_best_hyperparams
#             eighth_best_hyperparams = seventh_best_hyperparams
#             seventh_best_hyperparams = sixth_best_hyperparams
#             sixth_best_hyperparams = fifth_best_hyperparams
#             fifth_best_hyperparams = fourth_best_hyperparams
#             fourth_best_hyperparams = third_best_hyperparams
#             third_best_hyperparams = second_best_hyperparams
#             second_best_hyperparams = best_hyperparams
#             best_hyperparams = {
#                 'hidden_layer_lengthscale': hidden_layer_lengthscale,
#                 'hidden_layer_outputscale': hidden_layer_outputscale,
#                 'last_layer_lengthscale': last_layer_lengthscale,
#                 'last_layer_outputscale': last_layer_outputscale,
#                 'likelihood_noise': likelihood_noise_value,
#             }
#
#             tenth_best_result = ninth_best_result
#             ninth_best_result = eighth_best_result
#             eighth_best_result = seventh_best_result
#             seventh_best_result = sixth_best_result
#             sixth_best_result = fifth_best_result
#             fifth_best_result = fourth_best_result
#             fourth_best_result = third_best_result
#             third_best_result = second_best_result
#             second_best_result = best_result
#             best_result = {
#                 'loss': loss.item(),
#                 'test rmse': test_rmse,
#                 'train rmse': train_rmse
#             }
#         # normalize the train x and test x
#         mean_norm_x, std_norm_x = train_x.mean(), train_x.std()
#         train_x = (train_x - mean_norm_x) / (std_norm_x)
#         test_x = (test_x - mean_norm_x) / (std_norm_x)
#
#         # normalize the train y and test y
#         mean_norm, std_norm = train_y.mean(), train_y.std()
#         train_y = (train_y - mean_norm) / (std_norm)
#         test_y = (test_y - mean_norm) / (std_norm)
#
#         train_y = 10 ** (train_y / 20) * 100000
#         test_y = 10 ** (test_y / 20) * 100000
#
#         mean_norm_decimal, std_norm_decimal = train_y.mean(), train_y.std()
#         train_y = (train_y - mean_norm_decimal) / (std_norm_decimal)
#         test_y = (test_y - mean_norm_decimal) / (std_norm_decimal)




# # grid searching and separate validation
# method = ['grid searching and separate validation']
# model = MultitaskDeepGP(train_x.shape)
# if torch.cuda.is_available():
#     model = model.cuda()
# grid_times = 0
# for hidden_layer_lengthscale in lengthscale:
#     for hidden_layer_outputscale in outputscale:
#         for last_layer_lengthscale in lengthscale:
#             for last_layer_outputscale in outputscale:
#                 for likelihood_noise_value in likelihood_noise_range:
#                     for time in range(repeat_time):
#                         grid_times = grid_times+1
#                         hypers = {
#                             'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
#                                 [[[hidden_layer_lengthscale, hidden_layer_lengthscale]]]).to(device),  # -3,
#                             'hidden_layer.covar_module.raw_outputscale': torch.tensor([hidden_layer_outputscale]).to(device),
#                             'last_layer.covar_module.raw_outputscale': torch.tensor([last_layer_outputscale]).to(device),
#                             # 0, 1, 0.5
#                             'last_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor(
#                                 [[[last_layer_lengthscale]]]).to(device),
#                             'likelihood.raw_task_noises': torch.tensor([likelihood_noise_value]).to(device),
#                             'likelihood.raw_noise': torch.tensor([likelihood_noise_value]).to(device),
#                         }
#                         model.initialize(**hypers)
#                         model.train()
#                         optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#                         mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0))).to(device)
#                         # num_epochs = 1 if smoke_test else 100
#                         # num_epochs = 1 if smoke_test else 2000 best
#                         epochs_iter = tqdm.tqdm(range(params_epoch), desc='Epoch')
#                         loss_set = np.array([])
#                         for i in epochs_iter:
#                             optimizer.zero_grad()
#                             output = model(train_x)
#                             loss = -mll(output, train_y)
#                             epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
#                             loss_set = np.append(loss_set, loss.item())
#                             loss.backward()
#                             optimizer.step()
#                         model.eval()
#                         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                             predictions, predictive_variance = model.predict(test_x.float())
#                         predictions = predictions * std_norm_decimal + mean_norm_decimal
#                         test_y = test_y * std_norm_decimal + mean_norm_decimal
#                         predictions = 20 * torch.log10(predictions / 100000)
#                         test_y = 20 * torch.log10(test_y / 100000)
#
#                         predictions = predictions * std_norm + mean_norm
#                         test_y = test_y * std_norm + mean_norm
#                         # print('noise_covar',noise_convar)
#                         # print(predictive_variance)
#                         for task in range(0, 1):
#                             # print(task)
#                             # print(predictions[:, task])
#
#                             test_rmse = torch.mean(torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
#                         model.eval()
#                         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                             predictions, predictive_variance = model.predict(train_x.float())
#                         predictions = predictions * std_norm_decimal + mean_norm_decimal
#                         train_y = train_y * std_norm_decimal + mean_norm_decimal
#                         predictions = 20 * torch.log10(predictions / 100000)
#                         train_y = 20 * torch.log10(train_y / 100000)
#
#                         predictions = predictions * std_norm + mean_norm
#                         train_y = train_y * std_norm + mean_norm
#                         for task in range(0, 1):
#                             # print(task)
#                             # print(predictions[:, task])
#                             train_rmse = torch.mean(torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
#                         print(grid_times,
#                               '. hidden_lengthscale: ',hidden_layer_lengthscale,
#                               '; hidden_outputscale: ', hidden_layer_outputscale,
#                               '; last_lengthscale: ',last_layer_lengthscale,
#                               '; last_outputscale: ',last_layer_outputscale,
#                               '; noise: ',likelihood_noise_value)
#                         print('    loss:', loss.item(),
#                               ': test rmse: ',test_rmse,
#                               '; train rmse: ', train_rmse)
#                         if train_rmse<best_train:
#                             tenth_best_train = ninth_best_train
#                             ninth_best_train = eighth_best_train
#                             eighth_best_train = seventh_best_train
#                             seventh_best_train = sixth_best_train
#                             sixth_best_train = fifth_best_train
#                             fifth_best_train = fourth_best_train
#                             fourth_best_train = third_best_train
#                             third_best_train = second_best_train
#                             second_best_train = best_train
#                             best_train = train_rmse
#
#                             tenth_best_hyperparams = ninth_best_hyperparams
#                             ninth_best_hyperparams = eighth_best_hyperparams
#                             eighth_best_hyperparams = seventh_best_hyperparams
#                             seventh_best_hyperparams = sixth_best_hyperparams
#                             sixth_best_hyperparams = fifth_best_hyperparams
#                             fifth_best_hyperparams = fourth_best_hyperparams
#                             fourth_best_hyperparams = third_best_hyperparams
#                             third_best_hyperparams = second_best_hyperparams
#                             second_best_hyperparams = best_hyperparams
#                             best_hyperparams = {
#                                 'hidden_layer_lengthscale': hidden_layer_lengthscale,
#                                 'hidden_layer_outputscale': hidden_layer_outputscale,
#                                 'last_layer_lengthscale': last_layer_lengthscale,
#                                 'last_layer_outputscale': last_layer_outputscale,
#                                 'likelihood_noise': likelihood_noise_value,
#                             }
#
#                             tenth_best_result = ninth_best_result
#                             ninth_best_result = eighth_best_result
#                             eighth_best_result = seventh_best_result
#                             seventh_best_result = sixth_best_result
#                             sixth_best_result = fifth_best_result
#                             fifth_best_result = fourth_best_result
#                             fourth_best_result = third_best_result
#                             third_best_result = second_best_result
#                             second_best_result = best_result
#                             best_result = {
#                                 'loss': loss.item(),
#                                 'test rmse': test_rmse,
#                                 'train rmse': train_rmse
#                             }
#                         # normalize the train x and test x
#                         mean_norm_x, std_norm_x = train_x.mean(), train_x.std()
#                         train_x = (train_x - mean_norm_x) / (std_norm_x)
#                         test_x = (test_x - mean_norm_x) / (std_norm_x)
#
#                         # normalize the train y and test y
#                         mean_norm, std_norm = train_y.mean(), train_y.std()
#                         train_y = (train_y - mean_norm) / (std_norm)
#                         test_y = (test_y - mean_norm) / (std_norm)
#
#                         train_y = 10 ** (train_y / 20) * 100000
#                         test_y = 10 ** (test_y / 20) * 100000
#
#                         mean_norm_decimal, std_norm_decimal = train_y.mean(), train_y.std()
#                         train_y = (train_y - mean_norm_decimal) / (std_norm_decimal)
#                         test_y = (test_y - mean_norm_decimal) / (std_norm_decimal)

model = MultitaskDeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()

hypers = {
    'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor([[[best_hyperparams['hidden_layer_lengthscale'], best_hyperparams['hidden_layer_lengthscale']]]]).to(device), #-3,
    'hidden_layer.covar_module.raw_outputscale': torch.tensor([best_hyperparams['hidden_layer_outputscale']]).to(device),
    'last_layer.covar_module.raw_outputscale':torch.tensor([best_hyperparams['last_layer_outputscale']]).to(device), # 0, 1, 0.5
    'last_layer.covar_module.base_kernel.raw_lengthscale':torch.tensor([[[best_hyperparams['last_layer_lengthscale']]]]).to(device),
    'likelihood.raw_task_noises':torch.tensor([best_hyperparams['likelihood_noise']]).to(device),
    'likelihood.raw_noise':torch.tensor([best_hyperparams['likelihood_noise']]).to(device),
}


model.initialize(**hypers)
# model likelihood

print('likelihood.raw_task_noises ',model.likelihood.raw_task_noises)
print('likelihood.raw_noise ',model.likelihood.raw_noise )
# model.initialize(**hypers)
# print('hidden_layer.variational_strategy.inducing_points ',model.hidden_layer.variational_strategy.inducing_points)
# print('last_layer.variational_strategy.inducing_points ',model.last_layer.variational_strategy.inducing_points)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model,num_data=train_y.size(0))).to(device)
# num_epochs = 1 if smoke_test else 10000
# num_epochs = 1 if smoke_test else 2000 best
epochs_iter = tqdm.tqdm(range(optimal_epoch),desc='Epoch')
loss_set = np.array([])
for i in epochs_iter:
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
    loss_set = np.append(loss_set, loss.item())
    loss.backward()
    optimizer.step()
    # if i>3000 and loss.item() < 1.105:
    #     break
    # iteration_num=i

model.eval()
for param_name, param in model.named_parameters():

    if param_name == 'likelihood.raw_noise':
        noise_convar = param.item()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions, predictive_variance = model.predict(test_x.float())
    # print('prediction',predictions)
    # print('priedictions variance',predictive_variance)
    # print('model item',model.predict(test_x.float()))
    # mean = predictions.mean
    # lower, upper = predictions.confidence_region()
    lower = predictions - 1.96 * predictive_variance.sqrt()
    upper = predictions + 1.96 * predictive_variance.sqrt()
    # print('original lower',lower)
    # print('original upper', upper)


# print(predictions)
# print(predictions.size)
predictions = predictions*std_norm_decimal +mean_norm_decimal
test_y = test_y *std_norm_decimal + mean_norm_decimal
predictions = 20 * torch.log10(predictions/100000)
test_y = 20 * torch.log10(test_y/100000)

# print('std_norm_decimal',std_norm_decimal)
# print('mean_norm_decimal',mean_norm_decimal)
# print('std_norm',std_norm)
# print('mean_norm',mean_norm)

# lower and upper
lower = lower*std_norm_decimal +mean_norm_decimal
lower = 20 * torch.log10(lower/100000)
lower = lower*std_norm +mean_norm
upper = upper*std_norm_decimal +mean_norm_decimal
upper = 20 * torch.log10(upper/100000)
upper = upper*std_norm +mean_norm
# print('95% lower confident interval', lower)
# print('95% upper confident interval', upper)

# covar = covar *std_norm
# noise_convar = noise_convar*std_norm
predictions = predictions*std_norm +mean_norm
test_y = test_y *std_norm + mean_norm
# print('noise_covar',noise_convar)
# print(predictive_variance)
print('predicted test y',predictions[:].view(1,test_x.size(0)))
print('original test y',test_y[:].view(1,test_x.size(0)))

# test confident region
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(1, test_x.size(0)+1)
# print('x', x)
y = predictions[:].view(1,test_y.size(0)).squeeze().detach()
# print('y',y)
y = y.cpu().numpy()
# print('y',y)
ax.plot(x, y, 'b')
ax.plot(x, test_y.cpu().view(1,test_x.size(0)).squeeze().detach().numpy(), 'k*')
ax.fill_between(x, lower.cpu().view(1,test_x.size(0)).squeeze().detach().numpy(), upper.cpu().view(1,test_x.size(0)).squeeze().detach().numpy(), alpha=0.5)
ax.set_ylabel('RSS value')
ax.set_xlabel('data point')
ax.set_ylim([-105, -50])
ax.legend(['Mean','Observed Data',  'Confidence'], fontsize='x-small')
ax.set_title('DSVI DGP test confident interval-train %d point(s)/ test %d point(s)' % (len(train_y),len(test_y)))
# plt.show()


for task in range(0,1):
    # print(task)
    # print(predictions[:, task])

    rmse = torch.mean(torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
    print(task + 1, '. test RMSE:', rmse)
    # RMSE  less 3 and less 6 rate
    rmse_each = torch.pow(predictions[:, task] - test_y[:, task], 2).sqrt()
    # # print('rmse matrix',rmse_each)
    # less_1 = torch.sum(rmse_each < 1)
    # error_rate_1 = less_1 / test_y.size(0)
    # print('the test RMSE error larger than 1: ', error_rate_1.item(), '(error rate)', less_1.item(), '(number); ')
    # less_2 = torch.sum(rmse_each < 2)
    # error_rate_2 = less_2 / test_y.size(0)
    # print('the test RMSE error larger than 2: ', error_rate_2.item(), '(error rate)', less_2.item(), '(number); ')
    # less_3 = torch.sum(rmse_each < 3)
    # error_rate_3 = less_3 / test_y.size(0)
    # print('the test RMSE error larger than 3: ', error_rate_3.item(), '(error rate)', less_3.item(), '(number); ')
    # less_6 = torch.sum(rmse_each < 6)
    # error_rate_6 = less_6 / test_y.size(0)
    # print('the test RMSE error larger than 6: ', error_rate_6.item(), '(error rate)', less_6.item(), '(number); ')
    # less_10 = torch.sum(rmse_each < 10)
    # error_rate_10 = less_10 / test_y.size(0)
    # print('the test RMSE error larger than 10: ', error_rate_10.item(), '(error rate)', less_10.item(), '(number); ')
    # NLPD
    # term_1 = 0.5 * torch.log10(torch.mul(y_variance[:, task], 2 * math.pi))
    # term_2 = torch.sub(test_y[:, task], predictions[:, task]).pow(2).mul(1 / (2 * y_variance[:, task]))
    # final_nlpd = term_1 + term_2
    # mean_nlpd = final_nlpd.mean()
    # # MAE = torch.mean(torch.abs(preds.mean[:,task] - test_y[:,task]))
    # # max_y = torch.max(test_y[:,task])
    # # min_y = torch.min(test_y[:,task])
    # # nrmse = rmse / (max_y - min_y)


print('hidden_layer.covar_module.raw_outputscale, ',
      model.hidden_layer.covar_module.raw_outputscale
      )
print('hidden_layer.covar_module.base_kernel.raw_lengthscale, ',
      model.hidden_layer.covar_module.base_kernel.raw_lengthscale
      )
print('last_layer.covar_module.raw_outputscale ',
      model.last_layer.covar_module.raw_outputscale
      )
print('last_layer.covar_module.base_kernel.raw_lengthscale',
      model.last_layer.covar_module.base_kernel.raw_lengthscale
      )
print('likelihood.raw_task_noises ',model.likelihood.raw_task_noises )
print('likelihood.raw_noise',model.likelihood.raw_noise)
# training error
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions, predictive_variance = model.predict(train_x.float())
    # mean = predictions.mean
    # lower, upper = predictions.confidence_region()
    lower = predictions - 1.96 * predictive_variance.sqrt()
    upper = predictions + 1.96 * predictive_variance.sqrt()
# print(predictions)
# print(predictions.size)
# print('noise_covar',noise_convar)
print('One transmitter(y1)')
print('For Gaussian Process, 31 train / 2 test')
# print('noise_covar',noise_convar)
print('test error')
# print('noise_covar',noise_convar)
# print(predictive_variance)
print('final loss',loss.item())
# print('iteration num', iteration_num)
predictions = predictions*std_norm_decimal +mean_norm_decimal
train_y = train_y *std_norm_decimal + mean_norm_decimal
predictions = 20 * torch.log10(predictions/100000)
train_y = 20 * torch.log10(train_y/100000)
# covar = covar *std_norm
# noise_convar = noise_convar*std_norm
predictions = predictions*std_norm +mean_norm
train_y = train_y *std_norm + mean_norm
print('predicted train y',predictions[:].view(1,train_x.size(0)))
print('original train data',train_y[:].view(1,train_x.size(0)))

# lower and upper
# print('std_norm_decimal',std_norm_decimal)
# print('mean_norm_decimal',mean_norm_decimal)
# print('std_norm',std_norm)
# print('mean_norm',mean_norm)
lower = lower*std_norm_decimal +mean_norm_decimal
lower = 20 * torch.log10(lower/100000)
lower = lower*std_norm +mean_norm
upper = upper*std_norm_decimal +mean_norm_decimal
upper = 20 * torch.log10(upper/100000)
upper = upper*std_norm +mean_norm
# print('95% lower confident interval', lower)
# print('95% upper confident interval', upper)

for task in range(0,1):
    # print(task)
    # print(predictions[:, task])
    rmse = torch.mean(torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
    print(task + 1, '. train RMSE:', rmse)
    # # RMSE  less 3 and less 6 rate
    # rmse_each=torch.pow(predictions[:, task] - train_y[:, task], 2).sqrt()
    # # print('rmse matrix',rmse_each)
    # less_1 = torch.sum(rmse_each < 1)
    # error_rate_1 = less_1 / train_y.size(0)
    # print('the train RMSE error larger than 1: ', error_rate_1.item(), '(error rate)', less_1.item(), '(number); ')
    # less_2 = torch.sum(rmse_each < 2)
    # error_rate_2 = less_2 / train_y.size(0)
    # print('the train RMSE error larger than 2: ', error_rate_2.item(), '(error rate)', less_2.item(), '(number); ')
    # less_3 = torch.sum(rmse_each < 3)
    # error_rate_3 = less_3 / train_y.size(0)
    # print('the train RMSE error larger than 3: ', error_rate_3.item(), '(error rate)', less_3.item(), '(number); ')
    # less_6 = torch.sum(rmse_each < 6)
    # error_rate_6 = less_6 / train_y.size(0)
    # print('the train RMSE error larger than 6: ', error_rate_6.item(), '(error rate)', less_6.item(), '(number); ')
    # less_10 = torch.sum(rmse_each < 10)
    # error_rate_10 = less_10 / train_y.size(0)
    # print('the train RMSE error larger than 10: ', error_rate_10.item(), '(error rate)', less_10.item(), '(number); ')
    # NLPD
    # term_1 = 0.5 * torch.log10(torch.mul(y_variance[:, task], 2 * math.pi))
    # term_2 = torch.sub(test_y[:, task], predictions[:, task]).pow(2).mul(1 / (2 * y_variance[:, task]))
    # final_nlpd = term_1 + term_2
    # mean_nlpd = final_nlpd.mean()
    # # MAE = torch.mean(torch.abs(preds.mean[:,task] - test_y[:,task]))
    # # max_y = torch.max(test_y[:,task])
    # # min_y = torch.min(test_y[:,task])
    # # nrmse = rmse / (max_y - min_y)

# train confident region
fig, ax = plt.subplots(figsize=(18, 4))
x = np.arange(1, train_x.size(0)+1)
# print('x', x)
y = predictions[:].view(1,train_y.size(0)).squeeze().detach()
# print('y',y)
y = y.cpu().numpy()
# print('y',y)
ax.plot(x, y, 'b')
ax.plot(x, train_y.cpu().view(1,train_x.size(0)).squeeze().detach().numpy(), 'k*')
ax.fill_between(x, lower.cpu().view(1,train_x.size(0)).squeeze().detach().numpy(), upper.cpu().view(1,train_x.size(0)).squeeze().detach().numpy(), alpha=0.5)
ax.set_ylabel('RSS value')
ax.set_xlabel('data point')
ax.set_ylim([-105, -50])
ax.legend(['Mean','Observed Data',  'Confidence'], fontsize='x-small')
ax.set_title('DSVI DGP train confident interval-train %d point(s)/ test %d point(s)' % (len(train_y),len(test_y)))
# plt.show()


# loss figure
x = np.arange(1, optimal_epoch+1)
y = loss_set
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, linewidth=1.0)
ax.set_ylabel('loss value')
ax.set_xlabel('iteration times')
ax.set_title('DSVI DGP loss-train %d point(s)/ test %d point(s)' % (len(train_y),len(test_y)))
# plt.show()

# show the best and the second-best parameter and third-best parameter and the fourth
print('method',method)
print('the best parameter')
print(
    '    hidden_lengthscale: ',best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',best_hyperparams['last_layer_outputscale'],
    '; noise: ',best_hyperparams['likelihood_noise'])
print('    loss:', best_result['loss'],
    ': test rmse: ', best_result['test rmse'],
    '; train rmse: ', best_result['train rmse'])

print('the second best parameter')
print(
    '    hidden_lengthscale: ',second_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', second_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',second_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',second_best_hyperparams['last_layer_outputscale'],
    '; noise: ',second_best_hyperparams['likelihood_noise'])
print('    loss:', second_best_result['loss'],
    ': test rmse: ', second_best_result['test rmse'],
    '; train rmse: ', second_best_result['train rmse'])

print('the third best parameter')
print(
    '    hidden_lengthscale: ',third_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', third_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',third_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',third_best_hyperparams['last_layer_outputscale'],
    '; noise: ',third_best_hyperparams['likelihood_noise'])
print('    loss:', third_best_result['loss'],
    ': test rmse: ', third_best_result['test rmse'],
    '; train rmse: ', third_best_result['train rmse'])

print('the fourth best parameter')
print(
    '    hidden_lengthscale: ',fourth_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', fourth_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',fourth_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',fourth_best_hyperparams['last_layer_outputscale'],
    '; noise: ',fourth_best_hyperparams['likelihood_noise'])
print('    loss:', fourth_best_result['loss'],
    ': test rmse: ', fourth_best_result['test rmse'],
    '; train rmse: ', fourth_best_result['train rmse'])

print('the fifth best parameter')
print(
    '    hidden_lengthscale: ',fifth_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', fifth_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',fifth_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',fifth_best_hyperparams['last_layer_outputscale'],
    '; noise: ',fifth_best_hyperparams['likelihood_noise'])
print('    loss:', fifth_best_result['loss'],
    ': test rmse: ', fifth_best_result['test rmse'],
    '; train rmse: ', fifth_best_result['train rmse'])

print('the sixth best parameter')
print(
    '    hidden_lengthscale: ',sixth_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', sixth_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',sixth_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',sixth_best_hyperparams['last_layer_outputscale'],
    '; noise: ',sixth_best_hyperparams['likelihood_noise'])
print('    loss:', sixth_best_result['loss'],
    ': test rmse: ', sixth_best_result['test rmse'],
    '; train rmse: ', sixth_best_result['train rmse'])

print('the seventh best parameter')
print(
    '    hidden_lengthscale: ',seventh_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', seventh_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',seventh_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',seventh_best_hyperparams['last_layer_outputscale'],
    '; noise: ',seventh_best_hyperparams['likelihood_noise'])
print('    loss:', seventh_best_result['loss'],
    ': test rmse: ', seventh_best_result['test rmse'],
    '; train rmse: ', seventh_best_result['train rmse'])

print('the eighth best parameter')
print(
    '    hidden_lengthscale: ',eighth_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', eighth_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',eighth_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',eighth_best_hyperparams['last_layer_outputscale'],
    '; noise: ',eighth_best_hyperparams['likelihood_noise'])
print('    loss:', eighth_best_result['loss'],
    ': test rmse: ', eighth_best_result['test rmse'],
    '; train rmse: ', eighth_best_result['train rmse'])

print('the ninth best parameter')
print(
    '    hidden_lengthscale: ',ninth_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', ninth_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',ninth_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',ninth_best_hyperparams['last_layer_outputscale'],
    '; noise: ',ninth_best_hyperparams['likelihood_noise'])
print('    loss:', ninth_best_result['loss'],
    ': test rmse: ', ninth_best_result['test rmse'],
    '; train rmse: ', ninth_best_result['train rmse'])

print('the tenth best parameter')
print(
    '    hidden_lengthscale: ',tenth_best_hyperparams['hidden_layer_lengthscale'],
    '; hidden_outputscale: ', tenth_best_hyperparams['hidden_layer_outputscale'],
    '; last_lengthscale: ',tenth_best_hyperparams['last_layer_lengthscale'],
    '; last_outputscale: ',tenth_best_hyperparams['last_layer_outputscale'],
    '; noise: ',tenth_best_hyperparams['likelihood_noise'])
print('    loss:', tenth_best_result['loss'],
    ': test rmse: ', tenth_best_result['test rmse'],
    '; train rmse: ', tenth_best_result['train rmse'])
