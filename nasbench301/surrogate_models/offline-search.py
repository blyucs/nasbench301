import json
import os
import time
from torch.autograd import Variable
import click
import matplotlib
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from nasbench301.surrogate_models import utils
from nasbench301.surrogate_models.ensemble import Ensemble
from nasbench301.surrogate_models.gnn.gnn_utils import NASBenchDataset, Patience
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
import torch.optim as optim
from search_model import Gumbel_Sample_Model,Search_Model,Temp_Scheduler
matplotlib.use('Agg')

@click.command()
@click.option('--model', type=click.Choice(list(utils.model_dict.keys())), default='gnn',
              help='which surrogate model to fit')
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
@click.option('--model_config_path', type=click.STRING, help='Leave None to use the default config.', default=None)
@click.option('--data_config_path', type=click.STRING, help='Path to config.json',
              default='surrogate_models/configs/data_configs/nb_301.json')
@click.option('--log_dir', type=click.STRING, help='Experiment directory', default='experiments/surrogate_models')
@click.option('--seed', type=click.INT, help='seed for numpy, python, pytorch', default=6)
# @click.option('--ensemble', help='wether to use an ensemble', default=False)
# @click.option('--data_splits_root', type=click.STRING, help='path to directory containing data splits', default=None)
def train_surrogate_model(model, nasbench_data, model_config_path, data_config_path, log_dir, seed):
    # Load config
    data_config = json.load(open(data_config_path, 'r'))

    # Create log directory
    log_dir = os.path.join(log_dir, model, 'search-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), seed))
    os.makedirs(log_dir)

    # Select model config to use
    if model_config_path is None:
        # Get model configspace
        model_configspace = utils.get_model_configspace(model)

        # Use default model config
        model_config = model_configspace.get_default_configuration().get_dictionary()
    else:
        model_config = json.load(open(model_config_path, 'r'))
    model_config['model'] = model

    # Instantiate surrogate model
    surrogate_model = utils.model_dict[model](data_root=nasbench_data, log_dir=log_dir, seed=seed,
                                              model_config=model_config, data_config=data_config)

    # random init a graph
    # Instantiate dataset
    dataset = NASBenchDataset(root=surrogate_model.data_root, model_config=surrogate_model.model_config, result_paths=surrogate_model.test_paths,
                              config_loader=surrogate_model.config_loader)
    # Train and validate the model on the available data
    # init_point =  dataset.get(0)
    # surrogate_model.model.eval()
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)
    # def getTensor(self, var, grad=False):
    #     return torch.Tensor([var], requires_grad=grad)
    learning_rate = 0.02
    for step, graph_batch in enumerate(dataloader):
        random_batch = graph_batch
        # break
    # mapping1 = Variable(random_batch.edge_attr.float(), requires_grad=True)
    # random_batch.edge_attr = mapping1.expand_as(random_batch.edge_attr)
    # random_batch.edge_attr = random_batch.edge_attr.float()
    # random_batch.edge_attr.requires_grad = True
    # random_batch.edge_attr = torch.nn.Parameter(random_batch.edge_attr,requires_grad=True)
    # random_batch.edge_attr = Variable(random_batch.edge_attr,requires_grad=True)
    # 直接设置requires_grad=True

    x, edge_index, edge_attr, batch = random_batch.x.long(), random_batch.edge_index, random_batch.edge_attr.long(), random_batch.batch

    # load_dir = '/home/lvbo/00_code/nasbench301/nasbench301/nb_models_0.9/gnn_gin_v0.9/1/gnn_gin/20200910-200749-1'
    # load_dir = '/home/lvbo/00_code/nasbench301/nasbench301/gnn_gin/20240202-140310-6'
    # load_dir = '/home/lvbo/00_code/nasbench301/nasbench301/EXP/gnn_gin/train-20240205-110137-6'
    load_dir = '/home/lvbo/00_code/nasbench301/nasbench301/EXP/gnn_gin/train-20240221-140058-6'   # new model
    surrogate_model.load(os.path.join(load_dir, 'surrogate_model.model'))

    # for param in surrogate_model.model.parameters():
    #     param.requires_grad = False
    #
    # Init the sample model
    sample_model = Gumbel_Sample_Model(edge_attr)
    # Init the search model
    search_model = Search_Model(surrogate_model, sample_model)
    surrogate_model.model.eval()
    # edge_grad = edge_attr.float().requires_grad_(True)
    # alpha
    # edge_grad = edge_attr.requires_grad_(True)
    optimizer_p2 = optim.Adam(search_model.sample_model.parameters(), lr=learning_rate)
    base_temp = 1.0
    min_temp = 0.03
    epochs = 500
    temp_scheduler = Temp_Scheduler(epochs, sample_model._temp, base_temp,
                                               temp_min=min_temp)
    sample_model.get_cur_arch_attr()

    for i in range(epochs):
        optimizer_p2.zero_grad()
        pred = search_model(x, edge_index, batch)
        # pred = search_model.sample_model()
        loss = -pred.sum()
        loss.backward(retain_graph = True)
        # 在梯度更新前手动将特定行的梯度设置为0
        search_model.sample_model.zero_grad_identity_edge()
        optimizer_p2.step()
        search_model.sample_model._temp = temp_scheduler.step()
        # print(search_model.sample_model._temp)
        print(pred.sum())

    # surrogate_model.train()

    # Test the model
    # if len(surrogate_model.test_paths) > 0:
    #     surrogate_model.test()

if __name__ == "__main__":
    train_surrogate_model()
