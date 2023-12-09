""""
Grad-CAM visualization
Support for moganet, poolformer, deit, resmlp, resnet, swin and convnext
Modifed from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

please install the following packages
`pip install grad-cam timm`

Example command:
python cam_image.py --use_cuda --image_path /path/to/image.JPEG --model moganet_tiny
"""
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

import models  # register_model for MogaNet
import timm
from timm.models import create_model


def reshape_transform_resmlp(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='Input image path')
    parser.add_argument(
        '--output_image_path',
        type=str,
        default=None,
        help='Output image path')
    parser.add_argument(
        '--model',
        type=str,
        default='moganet_tiny',
        help='model name')
    parser.add_argument(
        '--aug_smooth',
        action='store_true',
        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument(
        '--method',
        type=str, default='gradcam',
        choices=['gradcam', 'gradcam++',
                 'scorecam', 'xgradcam',
                 'ablationcam', 'eigencam',
                 'eigengradcam', 'layercam', 'fullgrad'],
        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.model == 'resize':
        model = torch.nn.Identity()
    elif 'moganet' in args.model:
        model = create_model(args.model, pretrained=True)
    else:
        model = getattr(timm.models, args.model)(pretrained=('resnet' not in args.model))
    if 'resnet' in args.model:
        # resnet load the model trianed with 600 epochs
        # for fair comparison, load the model trained with 300 epochs.
        rsb_300epoch_dict = {
            'resnet18': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth',
            'resnet34': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth',
            'resnet50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a2_0-a2746f79.pth',
            'resnet101': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a2_0-6edb36c7.pth',
            'resnet152': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a2_0-b4c6978f.pth',
        }
        checkpoint = torch.hub.load_state_dict_from_url(url=rsb_300epoch_dict[args.model], map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)

    reshape_transform = None
    if 'moganet' in args.model:
        target_layers = [model.norm4]
    elif 'poolformer' in args.model:
        target_layers = [model.network[-1]]
    elif 'resnet' in args.model:
        target_layers = [model.layer4[-1]]
    elif 'convnext' in args.model:
        target_layers = [model.stages[-1]]
    elif 'resmlp' in args.model:
        target_layers = [model.blocks[-1]]
        reshape_transform = reshape_transform_resmlp
    elif 'deit' in args.model:
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
    elif 'swin' in args.model:
        target_layers = [model.layers[-1].blocks[-1]]
        reshape_transform = reshape_transform_swin

    model.eval()
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # target_layers = [model.layer4]
    # import pdb; pdb.set_trace()
    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    
    img_path = args.image_path
    if args.image_path:
        img_path = args.image_path
    else:
        import requests
        image_url = 'https://user-images.githubusercontent.com/44519745/212287714-98ead823-86c7-49d2-a8d7-4d77a2ce23e4.JPEG'  # n02510455_3566
        # image_url = 'https://user-images.githubusercontent.com/44519745/212287994-a957bdc0-d23c-48d8-a351-6799ce84182e.JPEG'  # n02165456_2300
        # image_url = 'https://user-images.githubusercontent.com/44519745/212288107-c5232dd8-ea67-4e8e-bdae-67c4f3033911.JPEG'  # n01664065_904
        # image_url = 'https://user-images.githubusercontent.com/44519745/212288207-e247e3d9-90fd-4bb2-883c-04598f46b66b.jpg'  # Blue_Jay_0009_62873
        img_path = image_url.split('/')[-1]
        if os.path.exists(img_path):
            img_data = requests.get(image_url).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)

    if args.output_image_path:
        save_name = args.output_image_path
    else:
        img_type = img_path.split('.')[-1]
        it_len = len(img_type)
        save_name = img_path.split('/')[-1][:-(it_len + 1)]
        save_name = save_name + '_' + args.model + '.' + img_type

    if args.model == 'resize':
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_name, img)
    else:
        img = Image.open(img_path)
        img = img.convert('RGB')
        image_transforms = Compose([Resize(256),
                                    CenterCrop(224)])
        format_transforms = Compose([ToTensor(),
                                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = image_transforms(img)
        input_tensor = format_transforms(img).unsqueeze(0)

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda,
                        reshape_transform=reshape_transform, 
                        ) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            rgb_img = np.array(img) / 255
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(save_name, cam_image)
