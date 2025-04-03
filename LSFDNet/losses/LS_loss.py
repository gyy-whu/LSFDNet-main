import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import MS_SSIMLoss, ssim_loss
from basicsr.utils import get_root_logger,tensor2img,imwrite
import core.metrics as Metrics
import numpy as np
from basicsr.utils.registry import LOSS_REGISTRY
# Parts of these codes are from: https://github.com/Linfeng-Tang/SeAFusion

def PSNR_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A)**2))/(m*n)
    MSE_BF = np.sum(np.sum((F - B)**2))/(m*n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255/np.sqrt(MSE))
    return PSNR

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis
        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)

        return loss_in, loss_grad

@LOSS_REGISTRY.register()
class Fusionloss_LS(nn.Module):
    def __init__(self):
        super(Fusionloss_LS, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, b,c ,image_vis, image_ir, generate_img, label, LW_th):
        alpha = b
        beta = c
        image_y = image_vis
        B, C, H, W = image_vis.shape
        image_ir = image_ir.expand(B, C, H, W)
        image_th = LW_th.expand(B, C, H, W)

        Lssim, L1loss = torch.zeros(B,10),torch.zeros(B,10)
        loss_MSE, loss_in, loss_grad, loss_label, loss_ss = torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B)
        ls,lin=torch.zeros(B),torch.zeros(B)

        x_in_mean = torch.add(image_y*0.5, image_th, alpha=0.5)
        x_in_mean_7 = torch.add(image_y*0.3, image_th, alpha=0.7)
        x_in_mean_max = torch.max(image_y, x_in_mean)
        # Gradient
        y_grad = self.sobelconv(image_y)
        x_in_mean_grad = self.sobelconv(x_in_mean)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, x_in_mean_grad)

        for b in range(B):
            loss_in[b] = 0.5 * F.l1_loss(generate_img[b], x_in_mean_max[b])
            loss_grad[b]  = F.l1_loss(generate_img_grad[b], x_grad_joint[b])
            loss_MSE[b] =  0.25 * self.mse_criterion(generate_img[b],image_y[b]) + 0.25 * self.mse_criterion(generate_img[b],image_ir[b]) 

        loss_global = alpha*(loss_in+loss_MSE) + (1-alpha)*loss_grad
        #label
        for b,batch_label in enumerate(label):
            for i,single_label in enumerate(batch_label): 
                if(single_label[1]==0 and single_label[2]==0 and single_label[3]==0 and single_label[4]== 0):
                    continue
                else:
                    object_vis = image_y[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_ir =  image_th[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_mean =  x_in_mean_7[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_generate = generate_img[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]

                    object_vis = torch.unsqueeze(object_vis,0)
                    object_ir =  torch.unsqueeze(object_ir,0)
                    object_mean = torch.unsqueeze(object_mean,0)
                    object_generate = torch.unsqueeze(object_generate,0)

                    object_in_max = torch.maximum(object_vis, object_mean)

                    ob_vis_grad = self.sobelconv(object_vis)
                    ob_generate_grad = self.sobelconv(object_generate)
                    ob_mean_grad = self.sobelconv(object_mean)
                    ob_grad_max = torch.maximum(ob_vis_grad, ob_mean_grad)

                    Lssim[b][i] = F.l1_loss(ob_generate_grad, ob_grad_max)
                    L1loss[b][i] = F.l1_loss(object_generate, object_in_max)

                
            exist = (Lssim[b] != 0) | (L1loss[b] != 0)
            if exist.sum()==0:
                loss_label[b] = 0
                ls[b]=0
                lin[b]=0
            else:
                ls[b]=Lssim[b].sum()/exist.sum()
                lin[b]=L1loss[b].sum()/exist.sum()
                loss_label[b] = (1-beta) * ls[b] + beta * lin[b]


        return loss_ss, loss_global, loss_label, loss_in, loss_grad,ls,lin


@LOSS_REGISTRY.register()
class Fusionloss_LS2(nn.Module):
    def __init__(self):
        super(Fusionloss_LS2, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, b,c ,image_vis, image_ir, generate_img, label, LW_th):
        alpha = b
        beta = c
        image_y = image_vis
        B, C, H, W = image_vis.shape
        image_ir = image_ir.expand(B, C, H, W)
        image_th = LW_th.expand(B, C, H, W)

        Lssim, L1loss = torch.zeros(B,10),torch.zeros(B,10)
        loss_MSE, loss_in, loss_grad, loss_label, loss_ss = torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B)
        ls,lin=torch.zeros(B),torch.zeros(B)

        x_in_mean = torch.add(image_y*0.5, image_th, alpha=0.5)
        x_in_mean_7 = torch.add(image_y*0.3, image_th, alpha=0.7)
        # x_max = torch.max(image_y, image_ir)
        x_in_max = torch.max(image_y, image_th)
        x_in_mean_max = torch.max(image_y, x_in_mean)
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_th)
        x_in_mean_grad = self.sobelconv(x_in_mean)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, x_in_mean_grad)

        # t=np.random.uniform(0, 1) 
        # grid_img = torch.cat((x_in_max,x_in_mean,x_in_mean_max,x_in_mean_7,generate_img), dim=0)
        # grid_img = tensor2img(grid_img, min_max=(-1, 1))
        # save_img_path = '/home/gyy/IRFusion-main/LSFDNet/test/pic_LS/{}_{}.png'.format(t,t)
        # imwrite(grid_img, save_img_path)




        for b in range(B):
            # loss_in[b] = F.l1_loss(generate_img[b], x_in_mean_max[b])
            # #loss_grad[b] = ssim_loss(generate_img[b].unsqueeze(0), x_in_max[b].unsqueeze(0), window_size=11)
            # loss_grad[b]  = F.l1_loss(generate_img_grad[b], x_in_mean_grad[b])

            loss_in[b] = F.l1_loss(generate_img[b], x_in_mean_max[b])
            loss_grad[b]  = F.l1_loss(generate_img_grad[b], x_grad_joint[b])
            # loss_MSE[b] =  0.5 * nn.MSELoss(generate_img[b],image_y) + 0.5 * nn.MSELoss(generate_img[b],image_ir) 
            


        # loss_global = alpha*loss_in + (1-alpha)*loss_grad + loss_MSE
        loss_global = alpha*loss_in + (1-alpha)*loss_grad
        #label
        for b,batch_label in enumerate(label):
            for i,single_label in enumerate(batch_label): 
                if(single_label[1]==0 and single_label[2]==0 and single_label[3]==0 and single_label[4]== 0):
                    continue
                else:
                    object_vis = image_y[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_ir =  image_th[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_mean =  x_in_mean_7[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_generate = generate_img[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]

                    object_vis = torch.unsqueeze(object_vis,0)
                    object_ir =  torch.unsqueeze(object_ir,0)
                    object_mean = torch.unsqueeze(object_mean,0)
                    object_generate = torch.unsqueeze(object_generate,0)

                    object_in_max = torch.maximum(object_vis, object_mean)

                    ob_vis_grad = self.sobelconv(object_vis)
                    ob_ir_grad = self.sobelconv(object_ir )
                    ob_generate_grad = self.sobelconv(object_generate)
                    ob_mean_grad = self.sobelconv(object_mean)
                    ob_grad_max = torch.maximum(ob_vis_grad, ob_mean_grad)

                    Lssim[b][i] = F.l1_loss(ob_generate_grad, ob_grad_max)
                    L1loss[b][i] = F.l1_loss(object_generate, object_in_max)

                
            exist = (Lssim[b] != 0) | (L1loss[b] != 0)
            if exist.sum()==0:
                loss_label[b] = 0
                ls[b]=0
                lin[b]=0
            else:
                ls[b]=Lssim[b].sum()/exist.sum()
                lin[b]=L1loss[b].sum()/exist.sum()
                loss_label[b] = (1-beta) * ls[b] + beta * lin[b]


        return loss_ss, loss_global, loss_label, loss_in, loss_grad,ls,lin

# @LOSS_REGISTRY.register()
# class Fusionloss_LS(nn.Module):
#     def __init__(self):
#         super(Fusionloss_LS, self).__init__()
#         self.sobelconv = Sobelxy()
#         self.mse_criterion = torch.nn.MSELoss()

#     def forward(self, b,c ,image_vis, image_ir, generate_img, label, LW_th):
#         alpha = b
#         beta = c
#         image_y = image_vis
#         B, C, H, W = image_vis.shape
#         image_ir = image_ir.expand(B, C, H, W)
#         image_th = LW_th.expand(B, C, H, W)

#         Lssim, L1loss = torch.zeros(B,10),torch.zeros(B,10)
#         loss_in, loss_grad, loss_label, loss_ss = torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B)
#         ls,lin=torch.zeros(B),torch.zeros(B)

#         x_in_mean = torch.add(image_y*0.5, image_th, alpha=0.5)
#         x_max = torch.max(image_y, image_ir)
#         x_in_max = torch.max(image_y, image_th)
#         # Gradient
#         y_grad = self.sobelconv(image_y)
#         ir_grad = self.sobelconv(image_th)
#         generate_img_grad = self.sobelconv(generate_img)
#         x_in_mean_grad = self.sobelconv(x_in_mean)
#         #B, C, K, H, W = y_grad.shape
#         #ir_grad = ir_grad.expand(B, C, K, H, W)
#         x_grad_joint = torch.maximum(y_grad, ir_grad)

#         i_grad = self.sobelconv(image_ir)
#         x_grad_ = torch.maximum(y_grad, i_grad)

#         # t=np.random.uniform(0, 1) 
#         # grid_img = torch.cat((x_in_max,x_in_max,x_in_max),dim=1)
#         # grid_img=torch.cat((grid_img,grid_img),dim=0)
#         # grid_img = Metrics.tensor2img(grid_img)
#         # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}.png'.format(t,t))
#         # grid_img = torch.cat((x_max,x_max,x_max),dim=1)
#         # grid_img=torch.cat((grid_img,grid_img),dim=0)
#         # grid_img = Metrics.tensor2img(grid_img)
#         # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}_0.png'.format(t,t))
#         # grid_img = torch.cat((x_grad_joint,x_grad_joint,x_grad_joint),dim=1)
#         # grid_img=torch.cat((grid_img,grid_img),dim=0)
#         # grid_img = Metrics.tensor2img(grid_img)
#         # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}_1.png'.format(t,t))
#         # grid_img = torch.cat((x_grad_,x_grad_,x_grad_),dim=1)
#         # grid_img=torch.cat((grid_img,grid_img),dim=0)
#         # grid_img = Metrics.tensor2img(grid_img)
#         # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}_2.png'.format(t,t))
#         # grid_img = torch.cat((LW_th,LW_th,LW_th),dim=1)
#         # grid_img=torch.cat((grid_img,grid_img),dim=0)
#         # grid_img = Metrics.tensor2img(grid_img)
#         # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}_3.png'.format(t,t))


#         for b in range(B):
#             loss_in[b] = F.l1_loss(generate_img[b], x_in_mean[b])
#             #loss_grad[b] = ssim_loss(generate_img[b].unsqueeze(0), x_in_max[b].unsqueeze(0), window_size=11)
#             loss_grad[b]  = F.l1_loss(generate_img_grad[b], x_in_mean_grad[b])


#         loss_global = alpha*loss_in + (1-alpha)*loss_grad

#         #label
#         for b,batch_label in enumerate(label):
#             for i,single_label in enumerate(batch_label): 
#                 if(single_label[1]==0 and single_label[2]==0 and single_label[3]==0 and single_label[4]== 0):
#                     continue
#                 else:
#                     object_vis = image_y[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
#                     object_ir =  image_ir[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
#                     object_generate = generate_img[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]

#                     object_vis = torch.unsqueeze(object_vis,0)
#                     object_ir =  torch.unsqueeze(object_ir,0)
#                     object_generate = torch.unsqueeze(object_generate,0)

#                     object_in_max = torch.maximum(object_vis, object_ir)

#                     ob_vis_grad = self.sobelconv(object_vis)
#                     ob_ir_grad = self.sobelconv(object_ir )
#                     ob_generate_grad = self.sobelconv(object_generate)
#                     ob_grad_max = torch.max(ob_vis_grad, ob_ir_grad)

#                     Lssim[b][i] = F.l1_loss(ob_generate_grad, ob_grad_max)
#                     L1loss[b][i] = F.l1_loss(object_generate, object_in_max)


#                 # t=0
#                 # grid_img = torch.cat((object_vis,object_vis,object_vis),dim=1)
#                 # grid_img=torch.cat((grid_img,grid_img),dim=0)
#                 # grid_img = Metrics.tensor2img(grid_img)
#                 # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}.png'.format(t,t))

#                 # grid_img = torch.cat((object_ir,object_ir,object_ir),dim=1)
#                 # grid_img=torch.cat((grid_img,grid_img),dim=0)
#                 # grid_img = Metrics.tensor2img(grid_img)
#                 # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}.png'.format(t+1,t+1))

#                 # grid_img = torch.cat(( object_generate , object_generate , object_generate ),dim=1)
#                 # grid_img=torch.cat((grid_img,grid_img),dim=0)
#                 # grid_img = Metrics.tensor2img(grid_img)
#                 # Metrics.save_img(grid_img, './exp_vis/pic/{}_{}.png'.format(t+2,t+2))
                
#             exist = (Lssim[b] != 0) | (L1loss[b] != 0)
#             if exist.sum()==0:
#                 loss_label[b] = 0
#                 ls[b]=0
#                 lin[b]=0
#             else:
#                 ls[b]=Lssim[b].sum()/exist.sum()
#                 lin[b]=L1loss[b].sum()/exist.sum()
#                 loss_label[b] = (1-beta) * ls[b] + beta * lin[b]


#         return loss_ss, loss_global, loss_label, loss_in, loss_grad,ls,lin
    
