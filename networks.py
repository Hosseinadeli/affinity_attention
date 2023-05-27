"""
Loads model. 
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
import torchvision.models as models
from torchvision.models.vgg import vgg16

from torchvision.models._utils import IntermediateLayerGetter

import dino.vision_transformer as vits
#import moco.vits as vits_moco

def get_model(arch, patch_size, device):

    # Initialize model with pretraining
    url = None
    resnet_dilate = 1
    if "resnet" in arch:
        if resnet_dilate == 1:
            replace_stride_with_dilation = [False, False, False]
        elif resnet_dilate == 2:
            replace_stride_with_dilation = [False, False, True]
        elif resnet_dilate == 4:
            replace_stride_with_dilation = [False, True, True]

        if "imagenet" in arch:
            model = resnet50(
                pretrained=True,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        else:
            model = resnet50(
                pretrained=False,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
            
    elif "vgg16" in arch:
        if "imagenet" in arch:
            model = vgg16(pretrained=True)
        else:
            model = vgg16(pretrained=False)
            
            
    elif "detr" in arch:
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
    elif "dinov2" in arch:
        #                 if arch == 'dinov2_vits14':    
#                     url = "dinov2/dinov2_vits14/ddinov2_vits14_pretrain.pth" 
#                 elif arch == "dinov2_vitb14":
#                     url = "dinov2/dinov2_vits14/ddinov2_vitb14_pretrain.pth" 
#                 elif arch == 'dinov2_vitl14': 
#                     url = "dinov2/dinov2_vits14/ddinov2_vitl14_pretrain.pth" 
#                 elif arch == 'dinov2_vitg14':    
#                     url = "dinov2/dinov2_vits14/ddinov2_vitg14_pretrain.pth" 
                    
        model = torch.hub.load('facebookresearch/dinov2', arch)
        
    else:
              
        if "moco" in arch:
            if arch == "moco_vit_small" and patch_size == 16:
                url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
            elif arch == "moco_vit_base" and patch_size == 16:
                url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            model = vits.__dict__[arch](num_classes=0)
        elif "mae" in arch:
            if arch == "mae_vit_base" and patch_size == 16:
                url = "mae/visualize/mae_visualize_vit_base.pth"
            model = vits.__dict__[arch](num_classes=0)
        elif "vit" in arch: 
            if arch == "vit_small" and patch_size == 16:
                url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and patch_size == 8:
                url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" 
            elif arch == "vit_base" and patch_size == 16:
                url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and patch_size == 8:
                url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
                
    if url is not None:
        print(
            "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
        )
                        
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        strict_loading = False if "resnet" in arch else True
        if "moco" in arch:
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "mae" in arch:
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('decoder') or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
        elif "resnet" in arch:
            #state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('fc'): # or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=strict_loading)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )

#     elif arch == "dino_resnet50":
#         url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    
#     elif arch == "resnet50_recog":
#         # Load the pre-trained ResNet50 model
#         resnet50 = models.resnet50(pretrained=True)
#         # Create a feature extractor by removing the classification layers
#         model = torch.nn.Sequential(*list(resnet50.children())[:-4])
        
#     else:
#         raise NotImplementedError 

#     if url is not None:
#         print(
#             "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
#         )
        
# #         if "resnet" in arch:
            
        
        
#         )
#     else:
#         print(
#             "There is no reference weights available for this model => We use random weights."
#         )


        
    # If ResNet or VGG16 loose the last fully connected layer
    if "resnet" in arch:
        model = ResNet50Bottom(model, return_interm_layers=1)
    elif "vgg16" in arch:
        model = vgg16Bottom(model)
                    
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)
    return model



class ResNet50Bottom(nn.Module):
    # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
    def __init__(self, backbone, return_interm_layers):
        super(ResNet50Bottom, self).__init__()
        # Remove avgpool and fc layers
        #self.features = nn.Sequential(*list(original_model.children())[:-3])
        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
            
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        #x = self.features(x)
        xs = self.body(x)
        return xs


class vgg16Bottom(nn.Module):
    # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
    def __init__(self, original_model):
        super(vgg16Bottom, self).__init__()
        # Remove avgpool and the classifier
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        # Remove the last maxPool2d
        self.features = nn.Sequential(*list(self.features[0][:-1]))

    def forward(self, x):
        x = self.features(x)
        return x
