#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from ModelTest import model_val, model_test
from helper import Trainer
from tools.metrics import record


sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))



def baseline_train(runid, model,
                   model_name,
                   dataloader,
                   device,
                   logger,
                   cfg):
    print("start training...", flush=True)
    save_path = os.path.join('Result', cfg['model_name'], cfg['data']['freq'], 'ckpt')
    demand_scalar = dataloader['scalar_taxi']

    engine = Trainer(model,
                     base_lr=cfg['train']['base_lr'],
                     weight_decay=cfg['train']['weight_decay'],
                     milestones=cfg['train']['milestones'],
                     lr_decay_ratio=cfg['train']['lr_decay_ratio'],
                     min_learning_rate=cfg['train']['min_learning_rate'],
                     max_grad_norm=cfg['train']['max_grad_norm'],
                     num_for_target=cfg['data']['num_for_target'],
                     num_for_predict=cfg['data']['num_for_predict'],
                     scaler=demand_scalar,
                     device=device,
                     )

    if cfg['train']['load_initial']:
        best_mode_path = cfg['train']['best_mode']
        logger.info("loading {}".format(best_mode_path))
        save_dict = torch.load(best_mode_path)
        engine.model.load_state_dict(save_dict['model_state_dict'])
        logger.info('model load success! {}'.format(best_mode_path))

    else:
        logger.info('Start training from scratch!')
        save_dict = dict()

    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_epoch = -1
    stable_count = 0
    logger.info('begin_epoch: {}, total_epochs: {}, patient: {}, best_val_loss: {:.4f}'.
                format(begin_epoch, epochs, tolerance, best_val_loss))

    for epoch in range(begin_epoch, begin_epoch + epochs + 1):
        train_gen_mape = {}
        train_gen_rmse = {}
        train_gen_mae = {}

        train_gen_mae['demand'] = []
        train_gen_rmse['demand'] = []
        train_gen_mape['demand'] = []
        train_loss = []
        t1 = time.time()
        train_dataloder = dataloader['train']
        train_tqdm_loader = tqdm(enumerate(train_dataloder))

        for iter, (demand, pos) in train_tqdm_loader:

            tpos = pos[..., :3]
            if demand.shape[3]>2:
                demand = demand[:,:,:,4:6].to(device)
            else:
                demand = demand.to(device)
            tpos = tpos.to(device)

            metrics = engine.train(demand, tpos)

            record(train_gen_mae, train_gen_rmse, train_gen_mape, metrics[1], metrics[2], metrics[3], only_last=True)
            train_loss.append(metrics[0])

            # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2 - t1)

        s1 = time.time()
        valid_loss, mvalid_mae, mvalid_rmse, mvalid_mape, valid_outputs = model_val(runid,
                                                                                 engine=engine,
                                                                                 dataloader=dataloader,
                                                                                 device=device,
                                                                                 logger=logger,
                                                                                 cfg=cfg,
                                                                                 epoch=epoch)
        s2 = time.time()
        val_time.append(s2 - s1)

        mtest_loss, mtest_mae, mtest_rmse , mtest_mape, test_outputs = model_test(runid, engine, dataloader, device,
                                                                                 logger, cfg, mode='Train')
        mtrain_loss = np.mean(train_loss)

        mtrain_pred_demand_mae = np.mean(train_gen_mae['demand'])
        mtrain_pred_demand_mape = np.mean(train_gen_mape['demand'])
        mtrain_pred_demand_rmse = np.mean(train_gen_rmse['demand'])

        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)

        if (epoch - 1) % cfg['train']['print_every'] == 0:
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(epoch, (s2 - s1)))

            log = 'Epoch: {:03d}, Train Total Loss: {:.4f} \n' \
                  'Train Pred Demand MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
            logger.info(log.format(epoch, mtrain_loss,
                                   mtrain_pred_demand_mae, mtrain_pred_demand_rmse, mtrain_pred_demand_mape,
                                   ))
            log = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
            logger.info(log.format(epoch, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse))

        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            epoch_best = epoch
            stable_count = 0
            save_dict.update(model_state_dict=copy.deepcopy(engine.model.state_dict()),
                             epoch=epoch_best,
                             best_val_loss=best_val_loss,
                             optimizer_state_dict=copy.deepcopy(engine.optimizer.state_dict()))

            ckpt_name = "exp{:s}_epoch{:d}_Val_mae:{:.2f}_mape:{:.2f}_rmse:{:.2f}.pth". \
                format(model_name, epoch, mvalid_mae, mvalid_mape, mvalid_rmse)
            best_mode_path = os.path.join(save_path, ckpt_name)
            torch.save(save_dict, best_mode_path)
            logger.info(f'Better model at epoch {epoch_best} recorded.')
            logger.info('Best model is : {}'.format(best_mode_path))
            logger.info('\n')

        else:
            stable_count += 1
            if stable_count > tolerance:
                break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)

    logger.info("Training finished")
    logger.info("The valid loss on best model is {:.4f}, epoch:{:d}".format(round(his_loss[bestid], 4), epoch_best))


    logger.info('Start the model test phase........')
    logger.info("loading the best model for this training phase {}".format(best_mode_path))
    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['model_state_dict'])

    valid_loss, valid_mae, valid_rmse, valid_mape, valid_outputs = model_val(runid,
                                                                             engine=engine,
                                                                             dataloader=dataloader,
                                                                             device=device,
                                                                             logger=logger,
                                                                             cfg=cfg,
                                                                             epoch=epoch)

    mtest_loss, mtest_mae, mtest_rmse, mtest_mape, test_outputs = model_test(runid, engine, dataloader, device,
                                                                             logger, cfg, mode='Test')

    return valid_mae, valid_mape, valid_rmse, mtest_mae, mtest_mape, mtest_rmse



