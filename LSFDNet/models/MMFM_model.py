import logging
import functools
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
from os import path as osp
import numpy as np
from basicsr.utils import get_root_logger,tensor2img,imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
from core.Metric_fusion.eval_one_method import evaluation_one_method_fast
import core.weights_init
from tqdm import tqdm
#from ultralytics import YOLO

@MODEL_REGISTRY.register()
class MMFM(BaseModel):
    def __init__(self, opt):
        super(MMFM, self).__init__(opt)
        logger = get_root_logger()
        # define network and load pretrained models
        self.netfe_base =  build_network(opt['network_fe_base'])
        self.netfe_base = self.model_to_device(self.netfe_base)
        self.print_network(self.netfe_base)
        
        self.netfe_SW =  build_network(opt['network_fe_SW'])
        self.netfe_SW = self.model_to_device(self.netfe_SW)
        self.print_network(self.netfe_SW)

        self.netfe_LW =  build_network(opt['network_fe_LW'])
        self.netfe_LW = self.model_to_device(self.netfe_LW)
        self.print_network(self.netfe_LW)

        self.netMF_mulLayer =  build_network(opt['network_MF_mulLayer'])
        self.netMF_mulLayer = self.model_to_device(self.netMF_mulLayer)
        self.print_network(self.netMF_mulLayer)

        if isinstance(self.netfe_base, (DataParallel, DistributedDataParallel)):
            self.netfe_base = self.netfe_base.module
        else:
            self.netfe_base = self.netfe_base

        if isinstance(self.netfe_SW, (DataParallel, DistributedDataParallel)):
            self.netfe_SW = self.netfe_SW.module
        else:
            self.netfe_SW = self.netfe_SW
        
        if isinstance(self.netfe_LW, (DataParallel, DistributedDataParallel)):
            self.netfe_LW = self.netfe_LW.module
        else:
            self.netfe_LW = self.netfe_LW

        if isinstance(self.netMF_mulLayer, (DataParallel, DistributedDataParallel)):
            self.netMF_mulLayer = self.netMF_mulLayer.module
        else:
            self.netMF_mulLayer = self.netMF_mulLayer

        if self.is_train:
            self.init_training_settings()
        else:
            self.netfe_base.eval()
            self.netfe_SW.eval()
            self.netfe_LW.eval()
            self.netMF_mulLayer.eval()

        self.current_iter = 0
        self.gama = opt['train']["a"]
        self.b=opt['train']["b"]
        self.c=opt['train']["c"]
        self._initialize_weights()  

    def _initialize_weights(self):  
        logger = get_root_logger()
        weights_init = functools.partial(core.weights_init.weights_init_normal)
        self.netfe_base.apply(weights_init)
        self.netfe_SW.apply(weights_init)
        self.netfe_LW.apply(weights_init)
        self.netMF_mulLayer.apply(weights_init)
        logger.info(f"Initialize weights of model")
    
    def init_training_settings(self):
        self.netfe_base.train()
        self.netfe_SW.train()
        self.netfe_LW.train()
        self.netMF_mulLayer.train()
        self.loss_dict_all = OrderedDict() 
        self.loss_dict_all['loss_all'] = []
        self.loss_dict_all['loss_back'] = []
        self.loss_dict_all['loss_label'] = []
        self.loss_dict_all['loss_in'] = []
        self.loss_dict_all['loss_grad'] = []
        self.loss_dict_all['ls'] = []
        self.loss_dict_all['lin'] = []
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.loss_LS = build_loss(self.opt['train']['Loss_LS']).to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_netfe_base_params = list(self.netfe_base.parameters())
        optim_netfe_SW_params = list(self.netfe_SW.parameters())
        optim_netfe_LW_params = list(self.netfe_LW.parameters())
        optim_netMF_mulLayer_params = list(self.netMF_mulLayer.parameters())
        optim_params_g_netfe_base = [{  # add normal params first
            'params': optim_netfe_base_params,
            'lr': train_opt['optimizer']['lr']
        }]    
        optim_params_g_netfe_SW = [{  # add normal params first
            'params': optim_netfe_SW_params,
            'lr': train_opt['optimizer']['lr']
        }] 
        optim_params_g_netfe_LW = [{  # add normal params first
            'params': optim_netfe_LW_params,
            'lr': train_opt['optimizer']['lr']
        }] 
        optim_params_g_netMF_mulLayer = [{  # add normal params first
            'params': optim_netMF_mulLayer_params,
            'lr': train_opt['optimizer']['lr']
        }]     
        optim_type = train_opt['optimizer'].pop('type')
        lr = train_opt['optimizer']['lr']
        self.optimizer_g_netfe_base = self.get_optimizer(optim_type, optim_params_g_netfe_base, lr)
        self.optimizers.append(self.optimizer_g_netfe_base)
        self.optimizer_g_netfe_SW = self.get_optimizer(optim_type, optim_params_g_netfe_SW, lr)
        self.optimizers.append(self.optimizer_g_netfe_SW)
        self.optimizer_g_netfe_LW = self.get_optimizer(optim_type, optim_params_g_netfe_LW, lr)
        self.optimizers.append(self.optimizer_g_netfe_LW)
        self.optimizer_g_netMF_mulLayer = self.get_optimizer(optim_type, optim_params_g_netMF_mulLayer, lr)
        self.optimizers.append(self.optimizer_g_netMF_mulLayer)


    # Feeding all data to the DF model
    def feed_data(self, train_data):
        self.data = self.set_device(train_data[0])

    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        self.current_iter = current_iter
        self.optimizer_g_netfe_base.zero_grad()
        self.optimizer_g_netfe_SW.zero_grad()
        self.optimizer_g_netfe_LW.zero_grad()
        self.optimizer_g_netMF_mulLayer.zero_grad()
        loss_dict = OrderedDict() 
        self.feats_SW_base = self.netfe_base(self.data['SW'])
        self.feats_LW_base = self.netfe_base(self.data['LW'])
        self.feats_SW = self.netfe_SW(self.feats_SW_base)
        self.feats_LW = self.netfe_LW(self.feats_LW_base)
        self.pred_img = self.netMF_mulLayer(self.feats_SW, self.feats_LW)
        #results = self.detmodel.predict(source=self.data["ir_detect"], show=False, save=True, save_txt=True) # Display preds. Accepts all YOLO predict arguments
        loss_ss, loss_back, loss_label, loss_in,loss_grad, ls,lin = self.loss_LS(self.b,self.c,image_vis=self.data["SW"], image_ir=self.data["LW"],  generate_img=self.pred_img, label=self.data["label"], LW_th=self.data["LW_th"])
        for i in range(len(loss_label)):
            if loss_label[i] !=0 :
                loss_ss[i] = self.gama * loss_back[i] + (1-self.gama) * loss_label[i]
            else:
                loss_ss[i] = loss_back[i]
        loss_fs=loss_ss.mean()
        loss_fs.backward()
        self.optimizer_g_netfe_base.step()
        self.optimizer_g_netfe_SW.step()
        self.optimizer_g_netfe_LW.step()
        self.optimizer_g_netMF_mulLayer.step()

        loss_dict['loss_all'] = loss_ss
        loss_dict['loss_back'] = loss_back
        loss_dict['loss_label'] = loss_label
        loss_dict['loss_in'] = loss_in
        loss_dict['loss_grad'] = loss_grad
        loss_dict['ls'] = ls
        loss_dict['lin'] = lin
        loss_dict = self.set_device(loss_dict)
        self.log_dict = self.reduce_loss_dict(loss_dict)
        for name, value in self.log_dict.items():
            self.loss_dict_all[name].append(value)


    # Testing on given data
    def test(self):
        self.netfe_base.eval()
        self.netfe_SW.eval()
        self.netfe_LW.eval()
        self.netMF_mulLayer.eval()
        with torch.no_grad():
            self.feats_SW_base = self.netfe_base(self.data['SW'])
            self.feats_LW_base = self.netfe_base(self.data['LW'])
            self.feats_SW = self.netfe_SW(self.feats_SW_base)
            self.feats_LW = self.netfe_LW(self.feats_LW_base)
            self.pred_img = self.netMF_mulLayer(self.feats_SW, self.feats_LW)
        self.netfe_base.train()
        self.netfe_SW.train()
        self.netfe_LW.train()
        self.netMF_mulLayer.train()

        return self.pred_img
    
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    if key == 'label':
                        x[key] = item.to(self.device, dtype=torch.int)
                    else:
                        x[key] = item.to(self.device, dtype=torch.float)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device, dtype=torch.float)
        else:
            x = x.to(self.device, dtype=torch.float)
        return x
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = self.opt['datasets']['val'].get('type')
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # for idx, val_data in enumerate(dataloader):
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(val_data[1][0])[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # ori_size = val_data[0]['ori_size'].cuda().squeeze(0).cpu().numpy()
            # ori_width, ori_height = ori_size[0], ori_size[1]
            # padded_, padded_channels, padded_height, padded_width = visuals['pred_img'].shape  
            # # 判断尺寸是否一致  
            # if padded_width != ori_width or padded_height != ori_height:  
            #     # 计算裁剪位置  
            #     left = int((padded_width - ori_width) // 2)  
            #     top = int((padded_height - ori_height) // 2)  
            #     right = int((padded_width + ori_width) // 2)  
            #     bottom = int((padded_height + ori_height) // 2)  

            #     # 裁剪图片  
            #     cropped_tensor = visuals['pred_img'][:,:, top:bottom, left:right]  

            sr_img = tensor2img(visuals['pred_img'].detach(), min_max=(-1, 1))
            metric_data['img'] = sr_img
            # save_img_path1 = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
            #                             f'{img_name}_1.jpg')
            # save_img_path2 = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
            #                             f'{img_name}_2.jpg')
            # ori_img = tensor2img(val_data[0]['SW'].detach(), min_max=(-1, 1))
            # ori_img1 = tensor2img(val_data[0]['LW'].detach(), min_max=(-1, 1))
            # imwrite(ori_img, save_img_path1)
            # imwrite(ori_img1, save_img_path2)

            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                             f'{img_name}.jpg')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, current_iter,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, current_iter,
                                                 f'{img_name}.jpg')
                imwrite(sr_img, save_img_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            r_dir, f_name = os.path.split(save_img_path) 
            save_dir =  osp.join(self.opt['path']['visualization'], 'metric')
            os.makedirs(save_dir, exist_ok=True)
            # EN, SF, AG, SD, CC, SCD, MSE, PSNR, Qabf, Nabf = evaluation_one_method_fast(dataset_name='OSLSP', data_dir='/data/gyy', result_dir='/home/gyy/IRFusion-main/LSFDNet/test', save_dir='/home/gyy/IRFusion-main/LSFDNet/test/metric_C2FM.xlsx', Method='C2FM' , with_mean=True)
            metric_r = evaluation_one_method_fast(dataset_name='OSLSP_320', data_dir='/data/gyy', result_dir=r_dir, save_dir= save_dir+ f'/{current_iter}' + '_metric_C2FM.xlsx', Method='C2FM' , with_mean=True)
            metric_f=['EN', 'SF', 'AG', 'SD', 'CC', 'SCD', 'MSE', 'PSNR', 'Qabf', 'Nabf']
            for index, metric in enumerate(metric_f):
                self.metric_results[metric] = metric_r[index][0]
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
            log_str = f'Validation {dataset_name}\n'
            for metric, value in self.metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}'
                if hasattr(self, 'best_metric_results'):
                    log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                                f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
                log_str += '\n'

            logger = get_root_logger()
            logger.info(log_str)
            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    # Get current log
    def get_current_iter_log(self):
        #self.update_loss()
        return self.log_dict
    
    def get_current_log(self):
        for name, value in self.log_dict.items():
            self.log_dict[name] = np.average(self.loss_dict_all[name])
        visuals = self.get_current_visuals()
        grid_img = torch.cat((visuals['pred_img'].detach(),
                                    visuals['gt_SW'],
                                    visuals['gt_LW']), dim=0)
        grid_img = tensor2img(grid_img, min_max=(-1, 1))
        save_img_path = os.path.join(self.opt['path']['visualization'],'img_fused_iter_{}.jpg'.format(self.current_iter))
        imwrite(grid_img, save_img_path)
        return self.log_dict


    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_img'] = self.pred_img
        out_dict['gt_SW'] = self.data["SW"]
        out_dict['gt_LW'] = self.data["LW"]
        return out_dict
    
    def save(self, epoch, current_iter):
        self.save_network([self.netfe_base, self.netfe_SW, self.netfe_LW], 'net_fe_g', current_iter, param_key=['params_base','params_SW','params_LW'])
        self.save_network([self.netMF_mulLayer], 'net_MLFM_g', current_iter, param_key=['params_MLFM'])
        self.save_training_state(epoch, current_iter)