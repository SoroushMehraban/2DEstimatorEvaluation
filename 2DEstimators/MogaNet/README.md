# MogaNet
Here we put the code we used for estimation of 2D sequences of Human3.6M using MogaNet.
## usage
1. Download he model weights from [here](https://github.com/Westlake-AI/MogaNet/releases/download/moganet-pose-weights/moganet_b_coco_256x192.pth) and place it inside `2DEstimators/MogaNet/pose_estimation/demo`.
2. Download the Faster R-CNN for person detection from [here](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) and place it here at `2DEstimators/MogaNet/pose_estimation/demo`.
3. Run the following command:
```
cd 2DEstimators/MogaNet/pose_estimation/demo/
python top_down_video_demo_with_mmdet.py  \
       mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
       faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_b_coco_256x192.py \
       moganet_b_coco_256x192.pth \
       --video-path <PATH-TO-INPUT-VIDEOS> \
       --out-pose-root <PATH-TO-OUTPUT-VIDEOS>
```
where the `--video-path` receives path to a directory containing the mp4 files and `--out-pose-root` receives path to a directory to store all the estimated pose sequences.