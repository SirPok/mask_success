checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=11,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'faster_rcnn_r50_fpn_1x_2.pth'
resume_from = None
workflow = [('train', 1)]
