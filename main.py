# !/usr/bin/env python
# -*- coding:utf-8 -*-
from datetime import datetime
import sys
import os
from ModelTest import baseline_test
from ModelTrain import baseline_train

from models.TGCSRN import get_TGCSRN

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from config.config import *
from preprocess.datasets import load_dataset_NYC, load_dataset_BJ
from tools.utils import sym_adj, asym_adj
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == '__main__':

    data_name = 'NYC'

    config_file = 'config_{:s}.yaml'.format(data_name)
    config_filename = 'config/'+config_file
    with open(config_filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    seed = cfg['seed']
    if cfg['test_only'] == False:
        seed = 666
    expid = cfg['expid']
    torch.manual_seed(seed)
    base_path = cfg['base_path']
    dataset_name = cfg['dataset_name']
    dataset_path = os.path.join(base_path, dataset_name)

    log_path = os.path.join('Result', cfg['model_name'], cfg['data']['freq'], 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    save_path = os.path.join('Result', cfg['model_name'], cfg['data']['freq'], 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    log_dir = log_path
    log_level = 'INFO'
    log_name = 'info_' + datetime.now().strftime('%m-%d_%H:%M') + '_exp' + str(expid) + '.log'
    logger = get_logger(log_dir, __name__, log_name, level=log_level)

    with open(os.path.join(log_dir, config_file), 'w+') as _f:
        yaml.safe_dump(cfg, _f)

    logger.info(cfg)
    logger.info(dataset_path)
    logger.info(log_path)

    torch.set_num_threads(3)
    device = torch.device(cfg['device'])

    if 'NYC' in dataset_name:
        dataloader = load_dataset_NYC(dataset_path,
                                      cfg['data']['train_batch_size'],
                                      cfg['data']['val_batch_size'],
                                      cfg['data']['test_batch_size'],
                                      )
    elif 'BJ' in dataset_name:
        dataloader = load_dataset_BJ(dataset_path,
                                      cfg['data']['train_batch_size'],
                                      cfg['data']['val_batch_size'],
                                      cfg['data']['test_batch_size'],
                                      )

    geo_graph = np.load(os.path.join(base_path, 'graph/geo_adj.npy')).astype(np.float32)

    if cfg['model']['norm_graph'] == 'sym':
        norm_geo_graph = torch.tensor(sym_adj(geo_graph)).to(device)
    elif cfg['model']['norm_graph'] == 'asym':
        norm_geo_graph = torch.tensor(asym_adj(geo_graph)).to(device)
    else:
        norm_geo_graph = torch.tensor(geo_graph).to(device)

    static_norm_adjs = [norm_geo_graph]
    cluster_num = cfg['data']['poi_cluster']
    model_name = cfg['model_name']

    input_dim = cfg['model']['input_dim']
    hidden_dim = cfg['model']['hidden_dim']
    output_dim = cfg['model']['output_dim']
    num_nodes = cfg['data']['cluster_num']
    num_for_target = cfg['data']['num_for_target']
    num_for_predict = cfg['data']['num_for_predict']

    activation_type = cfg['model']['activation_type']
    fuse_type = cfg['model']['fuse_type']
    node_emb = cfg['model']['node_emb']
    gcn_depth = cfg['model']['gcn_depth']

    val_mae_list = []
    val_mape_list = []
    val_rmse_list = []
    mae_list = []
    mape_list = []
    rmse_list = []
    for i in range(cfg['runs']):
        if model_name == 'TGCSRN':
            model = get_TGCSRN(in_dim=input_dim,
                               out_dim=output_dim,
                               hidden_dim=hidden_dim,
                               num_for_predict=num_for_predict,
                               num_nodes=num_nodes,
                               activation_type=activation_type,
                               supports=static_norm_adjs,
                               cluster_num=cluster_num,
                               device=device,
                               gcn_depth=gcn_depth,
                               fuse_type=fuse_type,
                               node_emb=node_emb).to(device)
        logger.info(model_name)

        if cfg['test_only']:
            val_mae, val_mape, val_rmse, mae, mape, rmse = baseline_test(i,
                                                                         model,
                                                                         dataloader,
                                                                         device,
                                                                         logger,
                                                                         cfg)
        else:
            val_mae, val_mape, val_rmse, mae, mape, rmse = baseline_train(i,
                                                                          model,
                                                                          cfg['model_name'],
                                                                          dataloader,
                                                                          device,
                                                                          logger,
                                                                          cfg,
                                                                          )

        val_mae_list.append(val_mae)
        val_mape_list.append(val_mape)
        val_rmse_list.append(val_rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

    mae_list = np.array(mae_list)
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)

    amae = np.mean(mae_list, 0)
    amape = np.mean(mape_list, 0)
    armse = np.mean(rmse_list, 0)

    smae = np.std(mae_list, 0)
    smape = np.std(mape_list, 0)
    srmse = np.std(rmse_list, 0)

    logger.info('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(val_mae_list), np.mean(val_rmse_list), np.mean(val_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(val_mae_list), np.std(val_rmse_list), np.std(val_mape_list)))
    logger.info('\n\n')

    logger.info('Test\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(mae_list), np.std(rmse_list), np.std(mape_list)))
    logger.info('\n\n')
