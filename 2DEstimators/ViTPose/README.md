# ViTPose
Here we put the code we used for estimation of 2D sequences of Human3.6M using ViTPose.
## usage
1. We have used ViTPose-H variant in our experiments. Download the model weights from [official repository](https://github.com/ViTAE-Transformer/ViTPose). Change the name to `vitpose-h.pth`, and place them here at `2DEstimators/ViTPose`
2. Download the Faster R-CNN for person detection from [here](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) and place it here at `2DEstimators/ViTPose`
3. Run the demo as follows:
```
python top_down_video_demo_with_mmdet.py  \
       demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
       faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
       vitpose-h.pth \
       --video-path <PATH-TO-INPUT-VIDEOS> \
       --out-pose-root <PATH-TO-OUTPUT-VIDEOS>
```
where the `--video-path` receives path to a directory containing the mp4 files and `--out-pose-root` receives path to a directory to store all the estimated pose sequences.