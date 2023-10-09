#!/usr/bin/env python
# -*- coding: UTF-8 -*-



import torch
import numpy as np
import os
from tqdm import tqdm

from helper import Trainer
from tools.metrics import metric, record


def model_val(runid, engine, dataloader, device, logger, cfg, epoch):
    logger.info('Start validation phase.....')

    val_dataloder = dataloader['val']

    valid_loss_list = []
    valid_outputs_list = []

    valid_pred_mape = {}
    valid_pred_rmse = {}
    valid_pred_mae = {}

    valid_pred_mae['demand'] = []
    valid_pred_rmse['demand'] = []
    valid_pred_mape['demand'] = []

    val_tqdm_loader = tqdm(enumerate(val_dataloder))
    for iter, (demand, pos) in val_tqdm_loader:
        tpos = pos[..., :3]
        tpos = tpos.to(device)

        # restore taxi
        if demand.shape[3] > 2:
            demand = demand[:, :, :, 4:6].to(device)
        else:
            demand = demand.to(device)

        loss, mae, rmse, mape, predict = engine.eval(demand, tpos)

        record(valid_pred_mae, valid_pred_rmse, valid_pred_mape, mae, rmse, mape, only_last=True)
        valid_loss_list.append(loss)
        valid_outputs_list.append(predict)

    mval_loss = np.mean(valid_loss_list)

    mvalid_pred_demand_mae = np.mean(valid_pred_mae['demand'])
    mvalid_pred_demand_mape = np.mean(valid_pred_mape['demand'])
    mvalid_pred_demand_rmse = np.mean(valid_pred_rmse['demand'])

    predicts = torch.cat(valid_outputs_list, dim=0)

    log = 'Epoch: {:03d}, Valid Total Loss: {:.4f}\n' \
          'Valid Pred Demand MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(epoch, mval_loss,
                           mvalid_pred_demand_mae, mvalid_pred_demand_rmse, mvalid_pred_demand_mape,
                           ))



    return mval_loss, mvalid_pred_demand_mae, mvalid_pred_demand_rmse, mvalid_pred_demand_mape, predicts


def model_test(runid, engine, dataloader, device, logger, cfg, mode='Train'):
    logger.info('Start testing phase.....')

    test_dataloder = dataloader['test']
    engine.model.eval()

    test_loss_list = []
    test_pred_mape = {}
    test_pred_rmse = {}
    test_pred_mae = {}
    test_pred_mae['demand'] = []
    test_pred_rmse['demand'] = []
    test_pred_mape['demand'] = []
    test_outputs_list = []

    test_tqdm_loader = tqdm(enumerate(test_dataloder))

    for iter, (demand, pos) in test_tqdm_loader:

        tpos = pos[..., :3]
        tpos = tpos.to(device)
        if demand.shape[3] > 2:
            demand = demand[:, :, :, 4:6].to(device)
        else:
            demand = demand.to(device)
        loss, gen_mae, gen_rmse, gen_mape, predict = engine.eval(demand, tpos)

        test_loss_list.append(loss)
        record(test_pred_mae, test_pred_rmse, test_pred_mape, gen_mae, gen_rmse, gen_mape, only_last=True)
        test_outputs_list.append(predict)

    mtest_loss = np.mean(test_loss_list)
    mtest_pred_demand_mae = np.mean(test_pred_mae['demand'])
    mtest_pred_demand_mape = np.mean(test_pred_mape['demand'])
    mtest_pred_demand_rmse = np.mean(test_pred_rmse['demand'])

    predicts = torch.cat(test_outputs_list, dim=0)
    log = 'Test Total Loss: {:.4f}\n' \
          'Test Pred Demand MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(mtest_loss,
                           mtest_pred_demand_mae, mtest_pred_demand_rmse, mtest_pred_demand_mape,))

    return mtest_loss, mtest_pred_demand_mae, mtest_pred_demand_rmse, mtest_pred_demand_mape, predicts


def baseline_test(runid, model, dataloader, device, logger, cfg):

    demand_scalar = dataloader['scalar_taxi']

    engine = Trainer(
        model,
        base_lr=cfg['train']['base_lr'],
        weight_decay=cfg['train']['weight_decay'],
        milestones=cfg['train']['milestones'],
        lr_decay_ratio=cfg['train']['lr_decay_ratio'],
        min_learning_rate=cfg['train']['min_learning_rate'],
        max_grad_norm=cfg['train']['max_grad_norm'],
        num_for_target=cfg['data']['num_for_target'],
        num_for_predict=cfg['data']['num_for_predict'],
        scaler=demand_scalar,
        device=device
    )

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path)
    try:
        engine.model.load_state_dict(save_dict['model_state_dict'])
    except:
        engine.model.load_state_dict(save_dict)
    logger.info('model load success! {}'.format(best_mode_path))

    mtest_loss, mtest_mae, mtest_rmse, mtest_mape, predicts = model_test(runid, engine, dataloader, device, logger,
                                                                         cfg, mode='Test')
    return mtest_mae, mtest_mape, mtest_rmse, mtest_mae, mtest_mape, mtest_rmse