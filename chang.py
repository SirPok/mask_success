import torch   
pretrained_weights = torch.load('checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

num_class = 2
pretrained_weights['state_dict']['bbox_head.fc_cls.weight'].resize_(num_class, 1024)
pretrained_weights['state_dict']['bbox_head.fc_cls.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.fc_cls.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['bbox_head.fc_cls.bias'].resize_(num_class*4)

torch.save(pretrained_weights, "faster_rcnn_r50_fpn_1x_%d.pth"%num_class)