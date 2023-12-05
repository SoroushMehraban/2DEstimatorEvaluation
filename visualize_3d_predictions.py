import numpy as np
import wandb

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from common.camera import *
from common.model_poseformer import *

from common.loss import *
from time import time
from common.utils import *
from common.h36m_dataset import Human36mDataset
from common.data_utils import create_checkpoint_dir_if_not_exists, preprocess_3d_data, load_2d_data, \
                              verify_2d_3d_matching, normalize_2d_data,\
                              init_train_generator, init_test_generator
from common.model_utils import count_number_of_parameters

# define keypoint connections and colors for plotting
connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]
L, R = 'b', 'r'
joint_colors = [L, R, R, R, L, L, L, L, L, L, L, L, L, L, R, R, R]
connection_colors = [L, L, L, R, R, R, L, L, L, L, L, R, R, R, L, L]


def align_pred(predicted, target):
    assert predicted.shape == target.shape, f"Pred shape is {predicted.shape} while target is {target.shape}"

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    return predicted_aligned


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


def evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field,
             joints_left, joints_right, action_idx, camera_idx, frame_idx):
    
    with torch.no_grad():
        model_pos.eval()
        
        video = 0
        target_video = ((action_idx + 1) - 1) * 4 + (camera_idx + 1)

        # Evaluate on test set and return the predictions for a specific frame
        for _, batch, batch_2d in test_generator.next_epoch():
            
            video += 1
            
            if (video == target_video): 
                
                inputs_3d = torch.from_numpy(batch.astype('float32')) # [1, 2356, 17, 3]
                inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [1, 2358, 17, 2]

                # apply test-time-augmentation (following Videopose3d)
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                # convert size
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d) # [2356, 3, 17, 2] 
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
                # print(inputs_2d[0].shape) [27, 17, 2]
                
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d = inputs_3d.cuda()
    
                inputs_3d[:, :, 0] = 0

                predicted_3d_pos = model_pos(inputs_2d)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                            joints_right + joints_left]
                predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                keepdim=True)
                
                # 3D ground truth for the specified frame
                inputs_3d_np = inputs_3d.cpu().numpy()
                inputs_3d_frame_np = inputs_3d_np[frame_idx] # [1, 17, 3]
                
                # 3D prediction for the specified frame
                predicted_3d_pos_np = predicted_3d_pos.cpu().numpy()
                predicted_3d_pos_frame_np = predicted_3d_pos_np[frame_idx]
                
                del inputs_2d, inputs_2d_flip, inputs_3d, predicted_3d_pos, predicted_3d_pos_flip
                torch.cuda.empty_cache()
                
    return inputs_3d_frame_np, predicted_3d_pos_frame_np


def inference(data_2d_path, dataset_3d, subjects_test, pad, joints_left, 
               joints_right, receptive_field, args, chk_filename, action_idx, camera_idx, frame_idx):
    
    keypoints_2d, kps_left, kps_right, num_joints = load_2d_data(data_2d_path)
    verify_2d_3d_matching(keypoints_2d, dataset_3d)
    normalize_2d_data(keypoints_2d, dataset_3d)
        
    test_generator = init_test_generator(subjects_test, keypoints_2d, dataset_3d, pad,
                                         kps_left, kps_right, joints_left, joints_right)
        
    model_pos = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args)
        
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
            
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
        
    gt_3d_frame, pred_3d_frame = evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field,
                                          joints_left, joints_right, action_idx, camera_idx, frame_idx)
    
    gt_3d_frame_plot = np.squeeze(gt_3d_frame)
    pred_3d_frame_aligned = align_pred(pred_3d_frame, gt_3d_frame)
    pred_3d_frame_plot = np.squeeze(pred_3d_frame_aligned)

    return pred_3d_frame_plot, gt_3d_frame_plot


def visualize_differences(gt_3d, transpose, vitpose, moganet, pct, cpn):
    
    titles = ["Ground Truth", "TransPose", "MogaNet", "PCT", "VitPose", "CPN"]
    fig = plt.figure(figsize=(30,30))
    axes = [fig.add_subplot(161, projection='3d'), fig.add_subplot(162, projection='3d'), fig.add_subplot(163, projection='3d'), fig.add_subplot(164, projection='3d'), fig.add_subplot(165, projection='3d'), fig.add_subplot(166, projection='3d')]
    
    for ax in axes:
        ax.view_init(-89, -90)
    
    gt_min, transpose_min, vitpose_min, moganet_min, pct_min, cpn_min = np.min(gt_3d, axis=0), np.min(transpose, axis=0), np.min(vitpose, axis=0), np.min(moganet, axis=0), np.min(pct, axis=0), np.min(cpn, axis=0)
    gt_max, transpose_max, vitpose_max, moganet_max, pct_max, cpn_max = np.max(gt_3d, axis=0), np.max(transpose, axis=0), np.max(vitpose, axis=0), np.max(moganet, axis=0), np.max(pct, axis=0), np.max(cpn, axis=0)
    min_values = np.min([gt_min, transpose_min, vitpose_min, moganet_min, pct_min, cpn_min], axis=0)
    max_values = np.max([gt_max, transpose_max, vitpose_max, moganet_max, pct_max, cpn_max], axis=0)
    axis_size = max_values - min_values
    aspect_ratio = np.array(axis_size / np.min(axis_size), dtype=int)

    for i, sequence in enumerate([gt_3d, transpose, moganet, pct, vitpose, cpn]):
        axes[i].set_xlim3d([min_values[0], max_values[0]])
        axes[i].set_ylim3d([min_values[1], max_values[1]])
        axes[i].set_zlim3d([min_values[2], max_values[2]])
        axes[i].set_box_aspect(aspect_ratio)

        x, y, z = sequence[:, 0], sequence[:, 1], sequence[:, 2]

        for j, connection in enumerate(connections):
            start = sequence[connection[0], :]
            end = sequence[connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            axes[i].plot(xs, ys, zs, c=connection_colors[j])

            start_gt = gt_3d[connection[0], :]
            end_gt = gt_3d[connection[1], :]
            xs = [start_gt[0], end_gt[0]]
            ys = [start_gt[1], end_gt[1]]
            zs = [start_gt[2], end_gt[2]]
            axes[i].plot(xs, ys, zs, color='black', linewidth=3, alpha=0.2)

        axes[i].scatter(x, y, z, c=joint_colors)

        title = titles[i]
        axes[i].set_title(title)
        axes[i].xaxis.set_ticklabels([])
        axes[i].yaxis.set_ticklabels([])
        axes[i].zaxis.set_ticklabels([])

    plt.savefig('figs/3d_prediction_comparison.png')

    
def visualize_differences_merging(gt_3d, manual, weighted_avg, avg):
    
    titles = ["Ground Truth", "Average", "Weighted Average", "Manual"]
    fig = plt.figure(figsize=(30,30))
    axes = [fig.add_subplot(141, projection='3d'), fig.add_subplot(142, projection='3d'), fig.add_subplot(143, projection='3d'), fig.add_subplot(144, projection='3d')]
    
    for ax in axes:
        ax.view_init(-89, -90)
    
    gt_min, manual_min, weighted_avg_min, avg_min = np.min(gt_3d, axis=0), np.min(manual, axis=0), np.min(weighted_avg, axis=0), np.min(avg, axis=0)
    gt_max, manual_max, weighted_avg_max, avg_max = np.max(gt_3d, axis=0), np.max(manual, axis=0), np.max(weighted_avg, axis=0), np.max(avg, axis=0)
    min_values = np.min([gt_min, manual_min, weighted_avg_min, avg_min], axis=0)
    max_values = np.max([gt_max, manual_max, weighted_avg_max, avg_max], axis=0)
    axis_size = max_values - min_values
    aspect_ratio = np.array(axis_size / np.min(axis_size), dtype=int)

    for i, sequence in enumerate([gt_3d, avg, weighted_avg, manual]):
        axes[i].set_xlim3d([min_values[0], max_values[0]])
        axes[i].set_ylim3d([min_values[1], max_values[1]])
        axes[i].set_zlim3d([min_values[2], max_values[2]])
        axes[i].set_box_aspect(aspect_ratio)

        x, y, z = sequence[:, 0], sequence[:, 1], sequence[:, 2]

        for j, connection in enumerate(connections):
            start = sequence[connection[0], :]
            end = sequence[connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            axes[i].plot(xs, ys, zs, c=connection_colors[j])

            start_gt = gt_3d[connection[0], :]
            end_gt = gt_3d[connection[1], :]
            xs = [start_gt[0], end_gt[0]]
            ys = [start_gt[1], end_gt[1]]
            zs = [start_gt[2], end_gt[2]]
            axes[i].plot(xs, ys, zs, color='black', linewidth=3, alpha=0.2)

        axes[i].scatter(x, y, z, c=joint_colors)

        title = titles[i]
        axes[i].set_title(title)
        axes[i].xaxis.set_ticklabels([])
        axes[i].yaxis.set_ticklabels([])
        axes[i].zaxis.set_ticklabels([])

    plt.savefig('figs/3d_prediction_comparison_merging_strategy.png')
 
    
def main():
    
    subjects_test = ['S9'] # subjects for evaluation ('S9' or 'S11')
    action_idx = 27 # for'S9', better to choose from (0, 10, 27, 29); for S11, better to choose from(0, 1, 26, 28)
    camera_idx = 2  # camera_idx (0 - 3) as there are 4 cameras in total
    frame_idx = 654 # the total number of frame for each action is different, the safe range (0 - 1000)
    
    args = parse_args()
    create_checkpoint_dir_if_not_exists(args.checkpoint)
    receptive_field = args.number_of_frames
    pad = (receptive_field -1) // 2  # Padding on each side

    # Data 3D
    dataset_path = f'data/data_3d_h36m.npz'
    dataset_3d = Human36mDataset(dataset_path)
    preprocess_3d_data(dataset_3d)
    joints_left, joints_right = list(dataset_3d.skeleton().joints_left()), list(dataset_3d.skeleton().joints_right())

    # Data 2D generated by various estimators and poseformerv2 checkpoints trained by using these 2D data as input
    data_2d_paths = ['data/data_2d_h36m_transpose.npz', 'data/data_2d_h36m_vitpose.npz', 'data/data_2d_h36m_moganet.npz', \
                     'data/data_2d_h36m_pct.npz', 'data/data_2d_h36m_cpn.npz', 'data/data_2d_h36m_merge_average.npz', \
                     'data/data_2d_h36m_merge_weighted_average.npz', 'data/data_2d_h36m_merge_manual.npz']
    
    ckp_paths = ['checkpoint/poseformerv2-transpose.bin', 'checkpoint/poseformerv2-vitpose.bin', \
                 'checkpoint/poseformerv2-moganet.bin', 'checkpoint/poseformerv2-pct.bin', \
                 'checkpoint/poseformerv2-cpn.bin', 'checkpoint/poseformerv2-merge-average.bin', \
                 'checkpoint/poseformerv2-merge-weighted-average.bin', 'checkpoint/poseformerv2-merge-manual.bin']
    
    # Find 3D ground truth and prediction for the specified frame by inferencing variants of the poseformerv2 models 
    pred_3d_transpose_plot, gt_3d_plot = inference(data_2d_paths[0], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                                   receptive_field, args, ckp_paths[0], action_idx, camera_idx, frame_idx)
    
    pred_3d_vitpose_plot, _ = inference(data_2d_paths[1], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                        receptive_field, args, ckp_paths[1], action_idx, camera_idx, frame_idx)
    
    pred_3d_moganet_plot, _ = inference(data_2d_paths[2], dataset_3d, subjects_test, pad, joints_left, joints_right, 
                                        receptive_field, args, ckp_paths[2], action_idx, camera_idx, frame_idx)
    
    pred_3d_pct_plot, _ = inference(data_2d_paths[3], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                    receptive_field, args, ckp_paths[3], action_idx, camera_idx, frame_idx)
    
    pred_3d_cpn_plot, _ = inference(data_2d_paths[4], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                    receptive_field, args, ckp_paths[4], action_idx, camera_idx, frame_idx)

    pred_3d_merge_average_plot, _ = inference(data_2d_paths[5], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                              receptive_field, args, ckp_paths[5], action_idx, camera_idx, frame_idx)
    
    pred_3d_merge_waverage_plot, _ = inference(data_2d_paths[6], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                               receptive_field, args, ckp_paths[6], action_idx, camera_idx, frame_idx)
    
    pred_3d_merge_manual_plot, _ = inference(data_2d_paths[7], dataset_3d, subjects_test, pad, joints_left, joints_right,
                                             receptive_field, args, ckp_paths[7], action_idx, camera_idx, frame_idx)
    
    # Generate plots to visualize the differences 
    visualize_differences(gt_3d_plot, pred_3d_transpose_plot, pred_3d_vitpose_plot, 
                          pred_3d_moganet_plot, pred_3d_pct_plot, pred_3d_cpn_plot)
    
    visualize_differences_merging(gt_3d_plot, pred_3d_merge_average_plot, 
                                  pred_3d_merge_waverage_plot, pred_3d_merge_manual_plot)

    
if __name__ == "__main__":
    main()
     