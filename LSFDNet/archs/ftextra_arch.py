import sys
import os
import torch
from torch import nn
from typing import List
import cv2
import numpy as np
from skimage.io import imsave
import inspect
from scripts.guided_diffusion.script_util import create_model_and_diffusion
from basicsr.utils.registry import ARCH_REGISTRY

#device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

@ARCH_REGISTRY.register()
class FeatureExtractorDDPM(nn.Module):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, input_activations: bool, steps: List[int], blocks: List[int], **kwargs):
        super().__init__()
        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)

        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    @torch.no_grad()
    def forward(self, x, noise=None, pth=None):
        activations = []
        for t1 in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t1]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        #return activations,out
        return activations
    


# class FeatureExtractorDDPM1(FeatureExtractor):
#     ''' 
#     Wrapper to extract features from pretrained DDPMs.
            
#     :param steps: list of diffusion steps t.
#     :param blocks: list of the UNet decoder blocks.
#     '''
    
#     def __init__(self, steps: List[int], blocks: List[int], **kwargs):
#         super().__init__(**kwargs)
#         self.steps = steps
        
#         # Save decoder activations
#         for idx, block in enumerate(self.model.output_blocks):
#             if idx in blocks:
#                 block.register_forward_hook(self.save_hook)
#                 self.feature_blocks.append(block)

#     def _load_pretrained_model(self, model_path, **kwargs):
#         import inspect
#         import guided_diffusion.dist_util as dist_util
#         from guided_diffusion.script_util import create_model_and_diffusion

#         # Needed to pass only expected args to the function
#         argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
#         expected_args = {name: kwargs[name] for name in argnames}
#         self.model, self.diffusion = create_model_and_diffusion(**expected_args)
#         self.model.load_state_dict(
#             dist_util.load_state_dict(model_path, map_location="cpu")
#         )
#         self.model.to(dist_util.dev())
#         #self.model=self.set_device(self.model)
#         if kwargs['use_fp16']:
#             self.model.convert_to_fp16()
#         self.model.eval()

#     @torch.no_grad()
#     def forward(self, x, noise=None, pth=None):
#         activations = []
#         for t1 in self.steps:
#             # Compute x_t and run DDPM
#             t = torch.tensor([t1]).to(x.device)
#             noisy_x = self.diffusion.q_sample(x, t, noise=noise)
#             self.model(noisy_x, self.diffusion._scale_timesteps(t))

#             # Extract activations
#             for block in self.feature_blocks:
#                 activations.append(block.activations)
#                 block.activations = None

#         # Per-layer list of activations [N, C, H, W]
#         #return activations,out
#         return activations


#     @torch.no_grad()
#     def _forward(self, x, noise=None, pth=None):
#         activations = []
#         out=[]
#         x=x
#         pth=pth[0]

        
#         save_root = './exp_vis'
#         file_path = os.path.join(save_root, 'progress') 
#         os.makedirs(file_path) if not os.path.exists(file_path) else file_path

#         img = x.detach_()
#         temp_img= img.detach().cpu().squeeze().numpy()
#         temp_img=np.transpose(temp_img, (1,2,0))
#         #temp_img=cv2.cvtColor(temp_img,cv2.COLOR_RGB2YCrCb)[:,:,0]
#         temp_img=(temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))
#         temp_img=((temp_img)*255).astype('uint8')
#         imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_x")),temp_img)


#         temp_img1=temp_img
#         temp_img1=np.concatenate((temp_img1[:,:,0:1],temp_img1[:,:,0:1],temp_img1[:,:,0:1]),2)
#         imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_x_SW")),temp_img1)
#         temp_img1=temp_img
#         temp_img1=np.concatenate((temp_img1[:,:,1:2],temp_img1[:,:,1:2],temp_img1[:,:,1:2]),2)
#         imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_x_LW")),temp_img1)


#         for t1 in self.steps:
#             # Compute x_t and run DDPM
#             t = torch.tensor([t1]).to(x.device)
#             noisy_x = self.diffusion.q_sample(x, t, noise=noise)

#             img = noisy_x.detach_()
#             temp_img= img.detach().cpu().squeeze().numpy()
#             temp_img=np.transpose(temp_img, (1,2,0))
#             #temp_img=cv2.cvtColor(temp_img,cv2.COLOR_RGB2YCrCb)[:,:,0]
#             temp_img=(temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))
#             temp_img=((temp_img)*255).astype('uint8')
#             imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_noisy_x{str(t1)}")),temp_img)

#             #out = self.diffusion.p_sample(self.model, x, self.diffusion._scale_timesteps(t))
#             out = self.diffusion.p_sample(self.model, x, self.diffusion._scale_timesteps(t))

#             img = out['pred_xstart'].detach_()
#             temp_img= img.detach().cpu().squeeze().numpy()
#             temp_img=np.transpose(temp_img, (1,2,0))
#             #temp_img=cv2.cvtColor(temp_img,cv2.COLOR_RGB2YCrCb)[:,:,0]
#             temp_img=(temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))
#             temp_img=((temp_img)*255).astype('uint8')
#             imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_pred_xstart{str(t1)}")),temp_img)

#             temp_img1=temp_img
#             temp_img1=np.concatenate((temp_img1[:,:,0:1],temp_img1[:,:,0:1],temp_img1[:,:,0:1]),2)
#             imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_pred_xstart{str(t1)}_SW")),temp_img1)
#             temp_img1=temp_img
#             temp_img1=np.concatenate((temp_img1[:,:,1:2],temp_img1[:,:,1:2],temp_img1[:,:,1:2]),2)
#             imsave(os.path.join(file_path, "{}.png".format(f"x_{pth}_pred_xstart{str(t1)}_LW")),temp_img1)

#             # Extract activations
#             for block in self.feature_blocks:
#                 activations.append(block.activations)
#                 block.activations = None



#         # for t1 in self.steps:
#         #     # Compute x_t and run DDPM
#         #     t = torch.tensor([t1]).to(x.device)
#         #     noisy_x = self.diffusion.q_sample(x, t, noise=noise)
#         #     self.model(noisy_x, self.diffusion._scale_timesteps(t))

#         #     # Extract activations
#         #     for block in self.feature_blocks:
#         #         activations.append(block.activations)
#         #         block.activations = None

#         # Per-layer list of activations [N, C, H, W]
#         #return activations,out
#         return activations,out

