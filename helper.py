# !/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tools.metrics import metric_all
from tools.utils import StepLR2


class Trainer():
    def __init__(self,
                 model,
                 base_lr,
                 weight_decay,
                 milestones,
                 lr_decay_ratio,
                 min_learning_rate,
                 max_grad_norm,
                 num_for_target,
                 num_for_predict,
                 scaler,
                 device,
                 ):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.scheduler = StepLR2(optimizer=self.optimizer,
                                 milestones=milestones,
                                 gamma=lr_decay_ratio,
                                 min_lr=min_learning_rate)

        self.loss = nn.SmoothL1Loss(reduction='mean')
        self.scaler = scaler
        self.num_for_target = num_for_target
        self.num_for_predict = num_for_predict

    def train(self, input, tpos):
        self.model.train()
        self.optimizer.zero_grad()

        demand_input = self.scaler.transform(input[:, :, :self.num_for_predict, :])

        real_val = input[:, :, -self.num_for_target:, :]

        output = self.model(demand_input,
                            tpos[:, :self.num_for_predict, :],
                            )


        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val)
        total_loss = loss

        total_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mae, rmse, mape = metric_all([predict],
                                   [real_val])
        return total_loss.item(), mae, rmse, mape

    def eval(self, input, tpos):
        self.model.eval()
        with torch.no_grad():

            demand_input = self.scaler.transform(input[:, :, :self.num_for_predict, :])

            real_val = input[:, :, -self.num_for_target:, :]
            output = self.model(demand_input,
                               tpos[:, :self.num_for_predict, :],
                               )

            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real_val)
            total_loss = loss

            mae, rmse, mape = metric_all([predict],
                                         [real_val])

        return total_loss.item(), mae, rmse, mape, predict