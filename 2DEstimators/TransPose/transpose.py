import os
import torch
import cv2
import torchvision.transforms as transforms
import argparse
import numpy as np
import torchvision
import warnings 
import pickle

NUM_KPTS = 17

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye', 
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

warnings.filterwarnings('ignore')


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1]
                      + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx], [dy]])
        hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i, j] = dr[border: -border, border: -border].copy()
            hm[i, j] *= origin_max / np.max(hm[i, j])
    return hm


def get_final_preds(hm, center, scale, transform_back=True, test_blur_kernel=3):
    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]

    # post-processing
    hm = gaussian_blur(hm, test_blur_kernel)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            coords[n, p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    if transform_back:
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

    return preds, maxvals


def get_person_detection_box(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return None
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]
    pred_score = pred_score[:pred_t + 1]

    person_box = None
    highest_score = float('-inf')
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person' and pred_score[idx] > highest_score:
            person_box = box
            highest_score = pred_score[idx]

    return person_box


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0
    img_size = (256, 192)
    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, img_size)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(img_size[0]), int(img_size[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    model_input = model_input.cuda()
    pose_model = pose_model.cuda()
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, max_vals = get_final_preds(
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds, max_vals
    

def inference(image_bgr, box_model, model):
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    box_model.to(CTX)
    box_model.eval()

    img_dimensions = (256, 192)

    input = []
    image_rgb = image_bgr[:, :, [2, 1, 0]]
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
    input.append(img_tensor)

    pred_box = get_person_detection_box(box_model, input, threshold=0.5)
    if pred_box is None:
        return None
    center, scale = box_to_center_scale(pred_box, img_dimensions[0], img_dimensions[1])
    image_pose = image_rgb.copy()
    pose_preds, max_vals = get_pose_estimation_prediction(model, image_pose, center, scale)
    if pose_preds.shape[0] > 1:
        print("[INFO] Prediction for a single frame is more than one! selecting the first one...")
        pose_preds = pose_preds[0][None, ...]
        max_vals = max_vals[0][None, ...]

    keypoints = np.concatenate([pose_preds, max_vals], axis=-1)
    return keypoints


def main(args):
    
    model = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True)
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()

    print("\nStart Inference...", flush=True)
    
    os.makedirs(args.out_pose_root, exist_ok=True)
    for video in os.listdir(args.video_path):
        
        if os.path.exists(os.path.join(args.out_pose_root, f'{video.replace(".", "_", 1).split(".")[0]}_pose_sequence.npy')):
            print(f'Skipping {video}, already processed')
            continue
            
        vid_path = os.path.join(args.video_path, video)
        print("Processing:", vid_path)

        cap = cv2.VideoCapture(vid_path)

        frame = 0
        keypoints = {}
        while (cap.isOpened()):
            read_successfully, img = cap.read()
            if not read_successfully:
                break

            # pass the frame to the estimator, and get 2D keypoint (17, 3) for each frame
            kpt = inference(img, box_model, model)
            keypoints[frame] = kpt
            frame += 1

        with open(os.path.join(args.out_pose_root, f'{video.replace(".", "_", 1).split(".")[0]}_pose_sequence.npy'), "wb") as f:
                np.save(f, keypoints)
                
        cap.release()
    
    print("Inference Complete!\n")
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video-path', type=str, required=True, help='Video path')
    parser.add_argument('--out-pose-root',  required=True, help='Root of the output pose tracking file.')

    args = parser.parse_args()
    
    main(args)
