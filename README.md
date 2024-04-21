<div align="center">
<h3>Efficient Deformable Modeling Network for Multi-View 3D Object Detection</h3>
</div>

<div align="center">
  <img src="figs/Overall_architecture_new.png" width="800"/>
</div><br/>

## Getting Started

Please follow our documentation step by step. If you like our work, please recommend it to your colleagues and friends.

1. [**Environment Setup.**](./docs/setup.md)
2. [**Data Preparation.**](./docs/data_preparation.md)
3. [**Training and Inference.**](./docs/training_inference.md)


## Results on NuScenes Val Set.
| Model | Setting |Pretrain| Lr Schd | NDS| mAP|Config |
| :---: | :---: | :---: | :---: | :---:| :---: | :---:|
|StreamPETR| R18 | ImageNet | 60ep | 48.4 | 36.2 |-|
|RepPETR4D| R18 | ImageNet | 60ep | 50.0 | 37.8 |[config](projects/configs/RepPETR4D/repdetr4d_res18_706_bs16_seq_60e.py)|
|StreamPETR| R50 | [NuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 60ep | 54.5 |44.9 |-|
|RepPETR4D| R50 | [NuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 60ep | 55.1 |45.5 |[config](projects/configs/RepPETR4D/repdetr4d_res50_706_bs16_seq_60e.py)|


## Results on NuScenes Test Set.
| Model | Setting |Pretrain|NDS| mAP|
| :---: | :---: | :---: | :---: | :---:|
|StreamPETR| R50 | [NuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 56.3| 46.0 |
|RepPETR4D| R50 | [NuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth)| 56.7| 46.9 |

## Acknowledgements

We thank these great works and open-source codebases:

* 3D Detection. [MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [SOLOFusion](https://github.com/Divadi/SOLOFusion), [Sparse4D](https://github.com/linxuewu/Sparse4D), [StreamPETR](https://github.com/exiawsh/StreamPETR).
