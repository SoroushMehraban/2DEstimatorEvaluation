# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(        
        type='MogaNet_feat',
        arch="tiny",  # modify 'arch' for various architectures
        init_value=1e-5,
        drop_path_rate=0.1,
        stem_norm_cfg=norm_cfg,
        conv_norm_cfg=norm_cfg,
        out_indices=(0, 1, 2, 3),
    ),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128, 256],  # modify 'in_channels'
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
