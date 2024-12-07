## Attention suits in autonomous driving (AD)

This repository aims to clearly and concisely introduce the basic implementation of attention commonly used in autonomous driving.

- Attention, Multi-head Attention
- Deformable Attention
- Query, Positional Encoding
- Transformer Encoder
- ViT
- Lift-Splat-Shoot

## Testing

```
python filename.py

e.g.
python multi_head_attention.py
python lss.py
```

If you want to compute the model's parameter count and computational cost, you can run the following code

```
python model_sta.py
```

## Acknowledgement

- [Attention is all you need](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf) [NeurIPS 2017]
- [Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d](https://krmzyc-filecloud.oss-cn-beijing.aliyuncs.com/theory/Lift%2C%20Splat%2C%20Shoot%20Encoding%20Images%20From%20Arbitrary%20Camera%20Rigs%20by%20Implicitly%20Unprojecting%20to%203D.pdf) [ECCV 2020]
- [Deformable detr: Deformable transformers for end-to-end object detection](https://arxiv.org/pdf/2010.04159) [ICLR 2021]
- [An image is worth 16x16 words: Transformers for image recognition at scale](https://bibbase.org/service/mendeley/bfbbf840-4c42-3914-a463-19024f50b30c/file/264ac473-27b7-bd53-3963-f6a07df9b72e/Dosovitskiy_et_al___2021___An_Image_is_Worth_16x16_Words_Transformers_for_Im.pdf.pdf) [ICLR 2021]
- [Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers](https://krmzyc-filecloud.oss-cn-beijing.aliyuncs.com/theory/BEVFormer%20Learning%20Bird%27s-Eye-View%20Representation%20from%20Multi-Camera%20Images%20via%20Spatiotemporal%20Trans.pdf) [ECCV 2022]
- [Bevfusion: Multi-task multi-sensor fusion with unified bird's-eye view representation](https://arxiv.org/pdf/2205.13542) [ICRA 2023]
- [Tri-perspective view for vision-based 3d semantic occupancy prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Tri-Perspective_View_for_Vision-Based_3D_Semantic_Occupancy_Prediction_CVPR_2023_paper.pdf) [CVPR 2023]
- [Planning-oriented autonomous driving](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.pdf) [CVPR 2023]
- [Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_SurroundOcc_Multi-camera_3D_Occupancy_Prediction_for_Autonomous_Driving_ICCV_2023_paper.pdf) [ICCV 2023]