# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch
import copy
import sys

# sys.path.append('../..')
from torch.utils.data import Dataset, DataLoader
from tools.utils import StandardScaler

def load_dataset_NYC(dataset_dir,
                     train_batch_size,
                     valid_batch_size=None,
                     test_batch_size=None,
                     ):
    cat_data = np.load(dataset_dir, allow_pickle=True)
    all_data = {
        'train': {
            'x': np.concatenate((cat_data['train_x'].transpose((0, 2, 1, 3)),cat_data['train_target'].transpose((0, 2, 1, 3))),axis=2),               # [batch, node_num, time, dim]
            'x_time': np.concatenate((cat_data['train_x_time'],cat_data['train_target_time']),axis=1),
            'pos': cat_data['train_pos'],
        },
        'val': {
            'x': np.concatenate((cat_data['val_x'].transpose((0, 2, 1, 3)),cat_data['val_target'].transpose((0, 2, 1, 3))),axis=2),               # [batch, node_num, time, dim]
            'x_time': np.concatenate((cat_data['val_x_time'],cat_data['val_target_time']),axis=1),
            'pos': cat_data['val_pos'],
        },
        'test': {
            'x': np.concatenate((cat_data['test_x'].transpose((0, 2, 1, 3)),cat_data['test_target'].transpose((0, 2, 1, 3))),axis=2),               # [batch, node_num, time, dim]
            'x_time': np.concatenate((cat_data['test_x_time'],cat_data['test_target_time']),axis=1),
            'pos': cat_data['train_pos'],
        },
    }

    train_wyc = np.concatenate(all_data['train']['x'])

    scaler = StandardScaler(mean=train_wyc.mean(), std=train_wyc.std())

    train_dataset = traffic_demand_prediction_dataset(all_data['train']['x'],
                                                      all_data['train']['x_time'],
                                                      )

    val_dataset = traffic_demand_prediction_dataset(all_data['val']['x'],
                                                    all_data['val']['x_time'],
                                                    None,
                                                    )

    test_dataset = traffic_demand_prediction_dataset(all_data['test']['x'],
                                                     all_data['test']['x_time'],
                                                     None,
                                                     )

    dataloader = {}
    dataloader['train'] = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size)
    dataloader['val'] = DataLoader(dataset=val_dataset, shuffle=False, batch_size=valid_batch_size)
    dataloader['test'] = DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size)
    dataloader['scalar_taxi'] = scaler
    return dataloader


def load_dataset_BJ(dataset_dir,
                     train_batch_size,
                     valid_batch_size=None,
                     test_batch_size=None,
                     ):
    cat_data = np.load(dataset_dir, allow_pickle=True)
    all_data = {
        'test': {
            'x': cat_data['test_x'],               # [batch, node_num, time, dim]
            'x_time': cat_data['test_x_time'],
            'pos': cat_data['test_pos'],
        },
    }

    scaler = StandardScaler(mean=30.190640167502103,
                            std=46.437284494007535)

    test_dataset = traffic_demand_prediction_dataset(all_data['test']['x'],
                                                     all_data['test']['x_time'],
                                                     None,
                                                     )
    dataloader = {}
    dataloader['test'] = DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size)
    dataloader['scalar_taxi'] = scaler
    return dataloader


class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x, x_time, target_cl=None):
        self.x = torch.tensor(x).to(torch.float32)
        self.x_time = torch.tensor(x_time).to(torch.float32)
        if target_cl is not None:
            self.target_cl = torch.tensor(target_cl).to(torch.float32)
        else:
            self.target_cl = None

    def __getitem__(self, item):

        if self.target_cl is not None:
            return self.x[item, :, :],\
                   self.x_time[item]
        else:
            return self.x[item, :, :],\
                   self.x_time[item]
    def __len__(self):
        return self.x.shape[0]