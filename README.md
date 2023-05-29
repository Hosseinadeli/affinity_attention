# Affinity Attention

<img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/sample_model_outputs/70158.png" width = 750>


<!-- Publications:
## Citing our work

Please cite our  work by using the following BibTeX entry.

``` bibtex
@Misc{xFormers2022,
  author =       {Benjamin Lefaudeux and Francisco Massa and Diana Liskovich and Wenhan Xiong and Vittorio Caggiano and Sean Naren and Min Xu and Jieru Hu and Marta Tintore and Susan Zhang and Patrick Labatut and Daniel Haziza},
  title =        {xFormers: A modular and hackable Transformer modelling library},
  howpublished = {\url{https://github.com/facebookresearch/xformers}},
  year =         {2022}
}
``` -->

[Hossein Adeli](https://hosseinadeli.github.io/)
hossein.adelijelodar@gmail.com

## Affinity spread model

You can run the model using the code below. 

```bash
python main.py --arch "dinov2_vitb14" --patch_size 14 --which_features "q"  --coco2017_path "./data/coco" --calc_object_centric 1 --calc_spread 1 --aff_tau 0.8
```

Two output files are saved for each run. First, with the number of steps that the attention took to reach the second for each trial. Second, the True Postiive and False Positive rates of benchmarking how object-centric the representiona is.  

We also provide two standalone colab notebookes: 
1. [affinity_attention_spread_Transformer_demo](https://colab.research.google.com/drive/1maUewiNxNXvm_OxEj_GMzi4fiHboNXyQ?usp=sharing) shows how affiniyt spread can be used to spread attention in a an object with features from a self-supervised vision transformer. 

2. [affinity_attention_spread_CNN_demo](https://colab.research.google.com/drive/16Ba_7ZPb2ptPdkBvSyzXNN3p2IftpJny?usp=sharing) shows how affiniyt spread can be used to spread attention in a an object with features from a CNN. 

## Behavioral paradigm and datasets: 

<img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/human_behavior/exp_paradigm.png" width = 350>                                     <img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/human_behavior/sample_trials.png" width = 300>

Refer to the paper for the details of the behavrioal paradigm. 

This [google drive folder](https://drive.google.com/drive/folders/10uXagNPxcPWh6YZgpCmBP_czUmxxXKWG?usp=sharing) includes all the experimental images:
**display_images** : This folder contains the images used in the experiment. All selected from COCO 2017 validation set. 
**display_images_with dots** : This folder contains the images used in the experiment with the four versions of dot placements. 

**Behavioral dataset**:
datasets -> datasets_grouping -> test_data_groupiong.xls : This excel file contains the info for all the 1020 experimental trials. Each row specified one trial with these info:

['img_id',
 'img_name',
 'same_diff',
 'close_far',
 'first_dot_xy',
 'second_dot_xy',
 'same_object_anns_ind',
 'diff_object_anns_ind',
 'distance',
 'Img_shape']

**datasets -> datasets_grouping -> train_data_groupiong.xls** : We applied our dot placement algorithm to the COCO 2017 training set in order to select many more trials. If you intend to train your model on the same-different task use this file. The images are completely separate from the images used in the experiment. The file contains the same info with the same format as the test_data_grouping.xls. 

**datasets -> loaddata_g.py** : This file can read in the excel files for testing and training and give you a pytorch dataloader with the below function:  

```bash
fetch_dataloader(args, batch_size, train='train', shuffle=True)
```

The args needs to have these components: 
args.batch_size – set as 1 because images have different sizes 
args.resize – whether to resize the images - only in use when running detr 
args.coco2017_path  – path to coco, have to be downloaded separately 
args.dataset_grouping_dir  - this is the folder where the xls files are

**utils for display images and dot locations.ipynb** : Some useful stuff to play around the display images and plot them. 

**datasets -> datasets_grouping -> Human_grouping_data.xls** : This is the main file containing raw behavioral data. We have 72 subjects in total. 

['RECORDING_SESSION_LABEL',
 'Trial_Index_',
 'subj_id',
 'trial_id',
 'block_id',
 'trial_id_block',
 'trial_type',
 'image_id',
 'img_id',
 'image',
 'image_dims',
 'target',
 'same_diff',
 'close_far',
 'first_dot_xy',
 'second_dot_xy',
 'first_dot_to_image',
 'first_dot_to_second_dot',
 'soa',
 'same_button',
 'REACTION_TIME',
 'RT_EVENT_BUTTON_ID']

**preprocess behavioral data.ipynb** : Use this notebook to preprocess the behavioral data. The preprocessed files (test_data_grouping_with_all_beh.xls, test_data_grouping_with_mean_beh.xls, df_beh_c.xls) are in the datasets -> datasets_grouping folder so no need to run this file, unless you want to change something. 

**human behavior analyses.ipynb** : This notebook takes in the preprocessed files and analyzes the RTs and percentiles. It also calculates the subject-subject agreements and plots some sample trials with average RTs from subjects. 


## Evaluation 

<img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/results/ROC_curve_sorted_sim.png" width = 375><img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/results/model_human_rt_comp_sim.png" width = 300>

**model results analyses.ipynb**: This notebook inputs the output runs and can plot the ROC curves for the object-centric measure. It can also compare each model ouput with the subjecs.

The ntoebook also shows hwo to plot the output number of steps for a given run. 

<img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/results/model_steps_hist_q_8_supp.png" width = 475><img src="https://raw.githubusercontent.com/Hosseinadeli/affinity_attention/main/figures/results/model_steps_q_8_supp.png" width = 250>


<!-- ### Repo map

```bash
├── ops                         # Functional operators
    └ ...
├── components                  # Parts zoo, any of which can be used directly
│   ├── attention
│   │    └ ...                  # all the supported attentions
│   ├── feedforward             #
│   │    └ ...                  # all the supported feedforwards
│   ├── positional_embedding    #
│   │    └ ...                  # all the supported positional embeddings
│   ├── activations.py          #
│   └── multi_head_dispatch.py  # (optional) multihead wrap
|
├── benchmarks
│     └ ...                     # A lot of benchmarks that you can use to test some parts
└── triton
      └ ...                     # (optional) all the triton parts, requires triton + CUDA gpu
``` -->
## Credits

The following repositories were used, either in close to original form or as an inspiration:

1) [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) <br/>
2) [facebookresearch/dino](https://github.com/facebookresearch/dino) <br/>
3) [facebookresearch/mae](https://github.com/facebookresearch/mae) <br/>
4) [facebookresearch/detr](https://github.com/facebookresearch/detr) <br/>
5) [YangtaoWANG95/TokenCut](https://github.com/YangtaoWANG95/TokenCut) <br/>
6) [valeoai/LOST](https://github.com/valeoai/LOST) <br/>
7) [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) <br/>

