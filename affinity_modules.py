import torch
import numpy as np
import scipy
from scipy import ndimage

import torchvision.transforms as Tvision
import torch.nn.functional as F

def aff_features(model, img, args):
    
    device = args.device

    if "vit"  in args.arch:

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

        # Store the outputs of qkv layer from the last attention layer
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

        # Forward pass in the model
        attentions = model.get_intermediate_layers(img[None, :, :, :].to(device)) #get_intermediate_layers

        # Scaling factor
        scales = [args.patch_size, args.patch_size]

        # Dimensions
        nb_im = 1 #attentions.shape[0]  # Batch size
        #nh = 12 #attentions.shape[1]  # Number of heads
        nb_tokens = h_featmap*w_featmap + 1 #attentions[0].shape[1]  # Number of tokens 1+

        # Baseline: compute DINO segmentation technique proposed in the DINO paper
        # and select the biggest component
        if args.dinoseg:
            pred = dino_seg(attentions, (h_featmap, w_featmap), args.patch_size, head=args.dinoseg_head)
            pred = np.asarray(pred)
        else:
            # Extract the qkv features of the last attention layer
            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, args.nh, -1 // args.nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            if args.which_features == "A":
                A = attentions.mean(1)[0,1:,1:]

            else:
                # Modality selection
                if args.which_features == "k":
                    #feats = k[:, 1:, :]
                    feats = k
                elif args.which_features == "q":
                    #feats = q[:, 1:, :]
                    feats = q
                elif args.which_features == "v":
                    #feats = v[:, 1:, :]
                    feats = v

                cls_token = feats[0,0:1,:].cpu().numpy() 

                feats_v = feats[0,1:,:]

                feats_v = F.normalize(feats_v, p=2)
                A = (feats_v @ feats_v.transpose(1,0)) 

    elif "detr" in args.arch:

        init_image_size = img.shape

        # use lists to store the outputs via up-values
        conv_features, enc_output, enc_attn_weights, dec_attn_weights = [], [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].register_forward_hook(
                lambda self, input, output: enc_output.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # propagate through the model
        outputs = model(img[None, :, :, :].to(device))

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]

        h_featmap, w_featmap = conv_features['0'].tensors.shape[-2:]

        #scales = [init_image_size[1]/h_featmap, init_image_size[2]/w_featmap]

        scales = [coco_img_size[0]/h_featmap, coco_img_size[1]/w_featmap]

        features = torch.squeeze(enc_output[0])
        features = F.normalize(features, p=2)
        A = (features @ features.transpose(1,0)) 

    elif "resnet" in args.arch:

        init_image_size = img.shape

        # Load an image and preprocess it
        #image = Image.open("image.jpg")
        preprocess = Tvision.Compose([
            Tvision.ToPILImage(),
    #                 Tvision.Resize(224),
    #                 Tvision.CenterCrop(224),
            Tvision.ToTensor(),
            Tvision.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = preprocess(img)
        image = torch.unsqueeze(image, 0)

        # Extract features from the image using the feature extractor
        features_multi = model(image.to(device))

        features = features_multi['2']

        h_featmap = features.shape[2]
        w_featmap = features.shape[3]

        scales = [init_image_size[1]/h_featmap, init_image_size[2]/w_featmap]

        features = torch.movedim(features.flatten(2), 1, -1)[0]

        features = F.normalize(features, p=2)
        A = (features @ features.transpose(1,0)) 

        #args.dataset = 'pathfinder'

        # Pass the features through the last convolutional layer to get the final features
        # final_features = last_conv_layer(features)

        # Print the shape of the final features
    #             print(h_featmap, w_featmap)
    #             print(features.shape)

    else:
        raise ValueError("Unknown model.")
        
    return A.cpu().numpy(), h_featmap, w_featmap, scales             
        

def aff_spread(A, idxs, scales, args, return_intermediate = False):
    
    tau = args.aff_tau
    tau_step = args.aff_tau_step

    A_center = A[..., idxs[0][0],idxs[0][1]].copy()
    A_center = (A_center>tau)*1

    labeled_array, num_labels = ndimage.label(A_center)

    A_center[ labeled_array != labeled_array[idxs[0][0],idxs[0][1]] ] = 0

    A_slice_p = A_center.copy()
    A_slice_p_new = A_center.copy()

    obj_mask = np.zeros_like(A[:,:,idxs[0][0],idxs[0][1]])
    obj_masks = []
    max_num_steps = 21
    steps_to_t = max_num_steps

    for i in range(max_num_steps-1):

        obj_mask = obj_mask + A_slice_p #+ A_slice_p_new

        #print(np.max(obj_mask))

        obj_mask = obj_mask/np.max(obj_mask)

        if len(idxs) > 1:
            if (obj_mask[idxs[1][0],idxs[1][1]]) > .0 and steps_to_t==max_num_steps:
                steps_to_t = i

        if return_intermediate:
            obj_masks.append(obj_mask[None,:,:])
        # normalize obj_mask? 

        # pick the new slice based on the connectivity to the most recent slice
        # A_slice = A[:,:,np.nonzero(A_slice_p)[0],np.nonzero(A_slice_p)[1]]

        # pick the new slice based on the connectivity to the entire object mask
        #obj_mask_s= (obj_mask>0.7)*1
        A_slice = A[:,:,np.nonzero(obj_mask)[0],np.nonzero(obj_mask)[1]]

        obj_mask_w = obj_mask[np.nonzero(obj_mask)[0],np.nonzero(obj_mask)[1]]

        #print(obj_mask_w)
        #A_slice = A_slice.mean(-1)
        #obj_mask_w = np.ones((A_slice.shape[2]))

        #A_slice = A_slice*obj_mask_w

        A_slice = A_slice.mean(-1)
        A_slice = (A_slice>tau)*1

        labeled_array, num_labels = ndimage.label(A_slice)

        A_slice[labeled_array != labeled_array[idxs[0][0],idxs[0][1]]] = 0

        A_slice_p = A_slice.copy()

        A_slice_p_new = A_slice_p - (obj_mask>0.0)*1
        A_slice_p_new = ((A_slice_p_new>0.0)*1).copy() 

        mask_p_change = np.sum(A_slice_p_new) / np.sum((obj_mask>0.0)*1)

        if tau > 0.0:  # 0.05
            tau = tau-tau_step

    if return_intermediate:
        return np.concatenate(obj_masks, axis=0), steps_to_t

    return obj_mask
