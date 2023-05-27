"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""

# load required libraries & modules

import os
import argparse
import random
import pickle

import torch
import datetime
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFilter
from scipy import ndimage
import cv2
from scipy import ndimage
import scipy
import copy 
from skimage.draw import disk

import warnings
warnings.filterwarnings("ignore")

import torchvision
import torchvision.transforms as Tvision


invTrans = Tvision.Compose([ Tvision.Normalize(mean = [ 0., 0., 0. ],
                                   std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                             Tvision.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                   std = [ 1., 1., 1. ]),
                           ])


from networks import get_model
from datasets_main import ImageDataset, Dataset_m, bbox_iou

import torch.nn.functional as F
#from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
#from object_discovery import ncut 
#import matplotlib.pyplot as plt
import time

from datasets.loaddata_g import *
from affinity_modules import aff_features, aff_spread

from pycocotools.coco import COCO

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="dinov2_vitb14",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "dino_resnet50",

            "moco_vit_small",
            "moco_vit_base",

            "mae_vit_base",

            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",

            "imagenet_resnet50",
            "imagenet_vit"  # add this in
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=14, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="grouping",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k", "grouping"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )

    parser.add_argument(
        "--set",
        default="test",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="q",
        choices=["k", "q", "v", "A"],
        help="Which features to use",
    )
    
    parser.add_argument(
        "--aff_tau", default=0.8, type=float, help="The threshold for the affinity spread model."
    )
           
    parser.add_argument(
        "--aff_tau_step", default=0.02, type=float, help="The threshold step for the affinity spread model."
    )
    
    parser.add_argument(
        "--calc_spread", default=0, type=int, help="Whether to measures spread."
    )
    
    parser.add_argument(
        "--calc_object_centric", default=0, type=int, help="Whether to measures the object-centricness of the represenation."
    )
    
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    
    parser.add_argument(
        "--coco2017_path",
        type=str,
        default='../../data/coco',  
        help="the directory for coco"
    )
    
    parser.add_argument(
        "--dataset_grouping_dir",
        type=str,
        default='./datasets/dataset_grouping/',
        help="the directory for coco"
    )
    
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    parser.add_argument("--method", type=str, default='spread')

    args = parser.parse_args()

    args.image_path = None #'examples/151_459153.png'
    args.visualize = 'attn' 
    args.save_vis = 0

    args.batch_size = 1
    
    verbose = 0
    
    nh = 12

    if args.arch == 'dinov2_vits14':
        nh = 6
    elif args.arch == 'dinov2_vitl14': 
        nh = 16
    elif args.arch == 'dinov2_vitg14':    
        nh = 24

    args.nh = nh
    if args.arch == 'detr':
        args.resize = 1

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    if args.dataset == 'grouping':
        args.save_predictions = False
        args.no_evaluation = True
        args.method = 'spread'

        if args.set == 'test':
            dataType = 'val2017'
        else:
            dataType = 'train2017'

        annFile='{}/annotations/instances_{}.json'.format(args.coco2017_path,dataType)

        # initialize COCO api for instance annotations
        coco=COCO(annFile)

    print(args)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    #device = torch.device('cuda') 
    model = get_model(args.arch, args.patch_size, device)
    #model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    #model

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
        test_dataloader =   dataset.dataloader
    elif args.dataset == 'grouping':
        test_dataloader = fetch_dataloader(args, args.set, shuffle=False)

    else:
        dataset = Dataset_m(args.dataset, args.set, args.no_hard)
        test_dataloader =   dataset.dataloader

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with TokenCut 
        exp_name = f"{args.arch}_{args.set}"  #_{args.dataset}
        if "vit" in args.arch:
            exp_name += f"_{args.patch_size}_{args.which_features}"
            
    tau_setting = f"_{args.aff_tau}_{args.aff_tau_step}"

    print(f"Running on the dataset {args.dataset} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    if args.save_feat_dir is not None : 
        os.mkdir(args.save_feat_dir)


    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(test_dataloader))

    all_steps_to_t = []

    start_time = time.time() 

    pbar = tqdm(test_dataloader)

    tetas = 1-np.arange(0,21)/20
    all_precision_recall = np.zeros((1024, 2, len(tetas)))  #len(test_dataloader)
    ROC = np.zeros((1024, 2, len(tetas)))


    # for im_id, inp in enumerate(pbar):
    all_iou = []
    all_mask_c_change = []

    for sample_id, sample  in enumerate(pbar):

    #     if sample_id<20 or sample_id>200: # sample_id%2==1:
    #         continue 

       # if sample_id==1024: break

        #print('sample_id : {}'.format(sample_id))

        #if (sample_id % 4) !=0: continue

        #TODO find a better way to pass bakc multiple element from the dataloader
        if args.dataset == 'grouping':

            img, dots_coords, targets, df_t = sample
            img = img[0]
            im_name = df_t['img_name'][0].split('.')[0]

        else:
            img = sample[0]

        # ------------ IMAGE PROCESSING -------------------------------------------
            # Get the name of the image
            im_name = dataset.get_image_name(sample[1])
            # Pass in case of no gt boxes in the image
            if im_name is None:
                continue
    #     img = inp[0]


        init_image_size = img.shape


        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        h_featmap = img.shape[-2] // args.patch_size
        w_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(sample[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            A, h_featmap, w_featmap, scales = aff_features(model, img, args)

            if args.method == 'spread':

                #if args.dataset == 'grouping'

                #idxs = [[14, 14], [15, 18]] # [14, 14]

                idxs = torch.zeros((2,2)).int()

                coords = 0.5* torch.ones((2,2,2))

                if args.dataset == 'grouping':
                    coords = dots_coords['coords']

                    idxs[0][0] = int(torch.round(coords[0][0][1] * h_featmap))
                    idxs[0][1] = int(torch.round(coords[0][0][0] * w_featmap))
                    idxs[1][0] = int(torch.round(coords[0][1][1] * h_featmap))
                    idxs[1][1] = int(torch.round(coords[0][1][0] * w_featmap))

    #             pred, objects, foreground, seed , bins, eigenvector= ncut(feats, [h_featmap, w_featmap], scales, init_image_size, args.tau, args.eps, im_name=im_name, no_binary_graph=args.no_binary_graph)

    #             idxs = np.array([[int(seed / w_featmap), int(seed % w_featmap)]])

                A_re = A.reshape(h_featmap,w_featmap,h_featmap,w_featmap)

                aff_map_c = A_re[:,:,idxs[0][0],idxs[0][1]]
                aff_map_c_re = scipy.ndimage.zoom(aff_map_c, scales, order=0, mode='wrap')

                aff_map_p = A_re[:,:,idxs[1][0],idxs[1][1]]
                aff_map_p_re = scipy.ndimage.zoom(aff_map_p, scales, order=0, mode='wrap')

                if args.save_vis:
                    Image.fromarray(np.uint8(aff_map_p_re*255)).save(f'figures/affinity_maps_{idxs[1][0]}_{idxs[1][1]}.png')
                    #fig.savefig(f'figures/affinity_maps_{idxs[1][0]}_{idxs[1][1]}.png', bbox_inches='tight', dpi=300)


                if (args.calc_object_centric):

                    annIds = coco.getAnnIds(imgIds=int(df_t['img_id']))  #catIds=catIds, ,  iscrowd=None
                    anns = coco.loadAnns(annIds)

                    # get the mask for same and different objects 
                    ann_mask_s = coco.annToMask(anns[int(df_t['same_object_anns_ind'])])
                    ann_mask_d = coco.annToMask(anns[int(df_t['diff_object_anns_ind'])])

                    if df_t['same_diff'][0] == 'same':
                        ann_mask = ann_mask_s
                    else:
                        ann_mask = ann_mask_d

                    # normazlie affinity to be between 0 and 1 for the purpose of this analysis

                    aff_map_p_re = aff_map_p_re - np.min(aff_map_p_re)
                    aff_map_p_re = aff_map_p_re / np.max(aff_map_p_re)

                    for te in range(len(tetas)):

                        active_area = (aff_map_p_re[:ann_mask.shape[0], :ann_mask.shape[1]] > tetas[te]) * 1

                        tp = np.sum(active_area*ann_mask) #np.maximum(1,

                        fn = np.sum((1-active_area)*ann_mask) 

                        fp = np.sum(active_area*(1-ann_mask)) 

                        tn = np.sum((1-active_area)*(1-ann_mask)) 

                        tpr = tp / (tp+fn)
                        fpr = fp / (fp+tn)

                        ROC[sample_id, 0, te] = tpr  
                        ROC[sample_id, 1, te] = fpr
                    
                    np.save(f"{args.output_dir}/ROCs/{exp_name}.npy", ROC)

                if (args.calc_spread):

                    A_re = A_re - np.min(A_re)
                    A_re = A_re / np.max(A_re)


                    obj_masks_c, steps_to_t = aff_spread(A_re, idxs, scales, args, return_intermediate = True)
                    all_steps_to_t.append(steps_to_t)

                    obj_masks_p, steps_to_t = aff_spread(A_re, idxs[[1, 0]], scales, args, return_intermediate = True)


                    trial_iou = []
                    trial_mask_c_change = []

                    for mask_step in range(len(obj_masks_c)):

                        obj_mask_c= obj_masks_c[mask_step]
                        obj_mask_p = obj_masks_p[mask_step]

                        obj_mask_c= (obj_mask_c>0.0)*1 
                        obj_mask_p = (obj_mask_p>0.0)*1

                        #obj_mask = obj_mask/np.max(obj_mask)

                        obj_mask = ((obj_mask_c+obj_mask_p)>1.5)*1

                        obj_mask_iou = np.sum(obj_mask) / np.sum(((obj_mask_c+obj_mask_p)>0.5)*1)

                        trial_iou.append(obj_mask_iou)

                        if (mask_step > 5):

                            obj_masks_change = ((obj_masks_c[mask_step]>0.0)*1) - ((obj_masks_c[mask_step-5]>0.0)*1)
                            obj_masks_change = ((obj_masks_change>0.0)*1)

                            mask_c_change = np.sum(obj_masks_change) / np.sum((obj_masks_c[mask_step]>0.0)*1)
                            trial_mask_c_change.append(mask_c_change)


                    all_iou.append(trial_iou)
                    all_mask_c_change.append(trial_mask_c_change)

                np.save(f"{args.output_dir}/model_att_steps/{exp_name}{tau_setting}.npy", np.array(all_steps_to_t))
                
