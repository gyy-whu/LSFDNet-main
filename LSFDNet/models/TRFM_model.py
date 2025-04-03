import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.utils import get_root_logger,tensor2img,imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
import scripts.guided_diffusion.dist_util as dist_util

@MODEL_REGISTRY.register()
class TRFM(BaseModel):
    def __init__(self, opt):
        super(TRFM, self).__init__(opt)
        # define network and load pretrained models

        self.netTR =  build_network(opt['network_fusion_head_trans'])
        self.netTR = self.model_to_device(self.netTR)
        self.print_network(self.netTR)

        self.netft =  build_network(opt['network_ft_extra'])
        self.netft = self.model_to_device(self.netft)
        logger = get_root_logger()
        # self.print_network(self.netft)
        logger.info(f"Pretrained model is successfully loaded from {opt['path']['pretrain_network_DDPM']}")

        if isinstance(self.netTR, (DataParallel, DistributedDataParallel)):
            self.netTR = self.netTR.module
        else:
            self.netTR = self.netTR
        load_path = self.opt['path'].get('pretrain_network_TRFM', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', None)
            self.load_network(self.netTR, load_path, self.opt['path'].get('strict_load_g', True), param_key)


        load_path = self.opt['path'].get('pretrain_network_DDPM', None)
        self.netft.module.model.load_state_dict(
            dist_util.load_state_dict(load_path, map_location="cpu")
        )
        if opt['network_ft_extra']['use_fp16']:
            self.netft.module.model.convert_to_fp16()
        self.netft.module.model.eval()
        
        if self.is_train:
            self.init_training_settings()
        else:
            self.netTR.eval()
            self.netft.eval()

        self.current_iter = 0
        self.gama = opt['train']["a"]
        self.b=opt['train']["b"]
        self.c=opt['train']["c"]

    def init_training_settings(self):
        self.netTR.train()
        self.netft.train()
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

        optim_df_params = list(self.netTR.parameters())
        optim_params_g = [{  # add normal params first
            'params': optim_df_params,
            'lr': train_opt['optimizer']['lr']
        }]
        optim_type = train_opt['optimizer'].pop('type')
        lr = train_opt['optimizer']['lr']
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, train_data):
        self.data = self.set_device(train_data[0])
        self.feats = self.netft(self.data['img'], noise=None, pth=None)
        

    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        self.current_iter = current_iter
        self.optimizer_g.zero_grad()
        loss_dict = OrderedDict() 
        self.pred_img = self.netTR(self.feats)
        #results = self.detmodel.predict(source=self.data["ir_detect"], show=False, save=True, save_txt=True) # Display preds. Accepts all YOLO predict arguments
        loss_ss, loss_back, loss_label, loss_in,loss_grad, ls,lin = self.loss_LS(self.b,self.c,image_vis=self.data["SW"], image_ir=self.data["LW"],  generate_img=self.pred_img, label=self.data["label"], LW_th=self.data["LW_th"])
        for i in range(len(loss_label)):
            if loss_label[i] !=0 :
                loss_ss[i] = self.gama * loss_back[i] + (1-self.gama) * loss_label[i]
            else:
                loss_ss[i] = loss_back[i]
        loss_fs=loss_ss.mean()
        loss_fs.backward()
        self.optimizer_g.step()

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
        self.netTR.eval()
        with torch.no_grad():
            self.pred_img, fd = self.netTR(self.feats)
            loss_ss, loss_in, loss_grad, loss_label = self.loss_func(image_vis=self.data["SW"],
                                                image_ir=self.data["LW"],
                                                generate_img=self.pred_img,
                                                label=self.data["label"])
            loss_fs = loss_in + loss_grad
            self.loss_all.append(loss_fs.item())
            self.loss_in.append(loss_in.item())
            self.loss_grad.append(loss_grad.item())
        self.netTR.train()
        return fd
    
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
        save_img_path = os.path.join(self.opt['path']['visualization'],'img_fused_iter_{}.png'.format(self.current_iter))
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
        self.save_network([self.netTR], 'net_g', current_iter, param_key=['params'])
        self.save_training_state(epoch, current_iter)


    def update_loss(self):
        self.log_dict['l_all'] = np.average(self.loss_all)
        self.log_dict['l_in'] = np.average(self.loss_in)
        self.log_dict['l_grad'] = np.average(self.loss_grad)
        ###
        self.loss_all = []
        self.loss_in = []
        self.loss_grad = []

    def _update_metric(self):
        """
        update metric
        """
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=self.data['L'].detach().cpu().numpy())
        return current_score

    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            self.log_dict[k] = v
            # message += '%s: %.5f ' % (k, v)

