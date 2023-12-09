# PCT
Here we put the code we used for estimation of 2D sequences of Human3.6M using PCT.
## usage
1. Download PCT weights from [here](https://mailustceducn-my.sharepoint.com/personal/aa397601_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faa397601%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FCVPR23%2DPCT%2Fpct&ga=1) and put it inside `2DEstimators/PCT/weights/pct`. We have used `swin_base.pth` in this project.
2. Download the detection weights from [here](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) and rename it to `mmdet_pct_config.pth` and put it here at `2DEstimators/PCT`.
3. Run the following command:
```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python3 vis_tools/demo_img_with_mmdet.py \
        vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py \
        mmdet_pct_config.pth configs/pct_base_classifier.py \
        weights/pct/swin_base.pth \
        --video-path <PATH-TO-INPUT-VIDEOS> \
        --out-pose-root <PATH-TO-OUTPUT-VIDEOS>
```
where the `--video-path` receives path to a directory containing the mp4 files and `--out-pose-root` receives path to a directory to store all the estimated pose sequences.