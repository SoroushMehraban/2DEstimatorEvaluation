# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (get_track_id, inference_top_down_pose_model, process_mmdet_results, init_pose_model)
from mmpose.datasets import DatasetInfo

# try:
from mmdet.apis import inference_detector, init_detector
has_mmdet = True

import sys
sys.path.append('../../')
import models  # register_model for MogaNet


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, default='', help='Video root')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--out-pose-root',
        default='',
        help='Root of the output pose tracking file.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    print('Loading videos:', flush=True)
    for video in os.listdir(args.video_path):
        print(f'{video.replace(".", "_", 1).split(".")[0]}_pose_sequence.npy', flush=True)
        if os.path.exists(os.path.join(args.out_pose_root, f'{video.replace(".", "_", 1).split(".")[0]}_pose_sequence.npy')):
            print(f'Skipping {video}, already loaded', flush=True)
            continue
        print(f'Loading {video}', flush=True)
        vid_path = os.path.join(args.video_path, video)
        cap = cv2.VideoCapture(vid_path)
        assert cap.isOpened(), f'Failed to load video file {args.video_path}'
        print(f'{video} loaded', flush=True)

        if args.out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_video_root, exist_ok=True)
            save_out_video = True

        fps = cap.get(cv2.CAP_PROP_FPS)
        if save_out_video:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root,
                         f'vis_{video}'), fourcc,
                fps, size)

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        poses = []
        frame_count = 0
        pose_results = []
        next_id = 0
        while (cap.isOpened()):
            pose_results_last = pose_results
            flag, img = cap.read()
            if not flag:
                break
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, img)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names,
                frame=frame_count)
            
            # get track id for each person instance
            pose_results, next_id = get_track_id(
                pose_results,
                pose_results_last,
                next_id,
                use_oks=args.use_oks_tracking,
                tracking_thr=args.tracking_thr,
                use_one_euro=args.euro,
                fps=fps)

            frame_count += 1

            for i in range(len(pose_results)):
                poses.append(pose_results[i])

        os.makedirs(args.out_pose_root, exist_ok=True)	
        with open(os.path.join(args.out_pose_root, f'{video.replace(".", "_", 1).split(".")[0]}_pose_sequence.npy'), "wb") as f:
            np.save(f, poses)

        cap.release()
        '''if save_out_video:
            videoWriter.release()'''
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

