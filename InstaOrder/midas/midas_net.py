"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder
import torch.distributed as dist
import utils
from models import backbone


class InstaDepthNet_d(BaseModel):
    """Network for order prediction and disparity estimation"""
    def __init__(self, path=None, features=256, depth_num_classes=3, occ_num_classes=2, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(InstaDepthNet_d, self).__init__()
        use_pretrained = False if path is None else True
        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features,
                                                      use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )
        if path:
            self.load(path)
            
        # create network conv for order
        self.gdo_net = backbone.__dict__['resnet50_cls'](in_channels=2, num_classes=3)
        self.gdo_net.layer1 = nn.Sequential(self.gdo_net.conv1, self.gdo_net.bn1, self.gdo_net.relu,
                                            self.gdo_net.maxpool, self.gdo_net.layer1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, depth_num_classes)
        utils.init_weights(self.gdo_net, init_type='xavier')


#         # module for depth order
#         self.do_net = backbone.__dict__['resnet50_cls'](in_channels=2, num_classes=depth_num_classes)
#         self.do_net.layer1 = nn.Sequential(self.do_net.conv1, self.do_net.bn1, self.do_net.relu,
#                                             self.do_net.maxpool, self.do_net.layer1)
#         self.depth_avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.depth_fc = nn.Linear(2048, depth_num_classes)

#         utils.init_weights(self.do_net, init_type='xavier')

    def forward(self, img, mask1, mask2):
        """Forward pass.

        Args:
            input data (image, mask1, mask2)

        Returns:
            tensor: disparity, depth order between mask1 and mask2
        """

        layer_1 = self.pretrained.layer1(img) #[1, 256, 96, 96]
        layer_2 = self.pretrained.layer2(layer_1) #[1, 512, 48, 48]
        layer_3 = self.pretrained.layer3(layer_2) #[1, 1024, 24, 24]
        layer_4 = self.pretrained.layer4(layer_3) #[1, 2048, 12, 12]
 
        layer_1_rn = self.scratch.layer1_rn(layer_1) #[1, 256, 96, 96]
        layer_2_rn = self.scratch.layer2_rn(layer_2) #[1, 256, 48, 48]
        layer_3_rn = self.scratch.layer3_rn(layer_3) #[1, 256, 24, 24]
        layer_4_rn = self.scratch.layer4_rn(layer_4) #[1, 256, 12, 12]

        path_4 = self.scratch.refinenet4(layer_4_rn) #[1, 256, 24, 24]
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn) #[1, 256, 48, 48]
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn) #[1, 256, 96, 96]
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn) #[1, 256, 192, 192]
        disp = self.scratch.output_conv(path_1) #[1, 1, 384, 384]

        # predict depth order
        do_feat_1 = self.gdo_net.layer1(torch.cat([mask1, mask2], dim=1)) #[1, 256, 96, 96]
        do_feat_2 = self.gdo_net.layer2(do_feat_1 + layer_1)  #[1, 512, 48, 48]
        do_feat_3 = self.gdo_net.layer3(do_feat_2 + layer_2)  #[1, 1024, 24, 24]
        do_feat_4 = self.gdo_net.layer4(do_feat_3 + layer_3)  #[1, 2048, 12, 12]


        depth_order = self.avgpool(do_feat_4)
        depth_order = torch.flatten(depth_order, 1)
        depth_order = self.fc(depth_order)

        
        
#         # predict depth order
#         do_feat_1 = self.do_net.layer1(torch.cat([mask1, mask2], dim=1)) #[1, 256, 96, 96]
#         do_feat_2 = self.do_net.layer2(do_feat_1 + layer_1)  #[1, 512, 48, 48]
#         do_feat_3 = self.do_net.layer3(do_feat_2 + layer_2)  #[1, 1024, 24, 24]
#         do_feat_4 = self.do_net.layer4(do_feat_3 + layer_3)  #[1, 2048, 12, 12]

#         depth_order = self.depth_avgpool(do_feat_4)
#         depth_order = torch.flatten(depth_order, 1)
#         depth_order = self.depth_fc(depth_order)
        
        return torch.squeeze(disp, dim=1), depth_order, None


class InstaDepthNet_od(BaseModel):
    """Network for order prediction and disparity estimation"""
    def __init__(self, path=None, features=256, depth_num_classes=3, occ_num_classes=2, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(InstaDepthNet_od, self).__init__()
        use_pretrained = False if path is None else True
        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features,
                                                      use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )
        if path:
            self.load(path)


        # module for depth order
        self.do_net = backbone.__dict__['resnet50_cls'](in_channels=2, num_classes=depth_num_classes)
        self.do_net.layer1 = nn.Sequential(self.do_net.conv1, self.do_net.bn1, self.do_net.relu,
                                            self.do_net.maxpool, self.do_net.layer1)
        self.depth_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.depth_fc = nn.Linear(2048, depth_num_classes)


        # module for occlusion order
        self.oo_net = backbone.__dict__['resnet50_cls'](in_channels=2, num_classes=occ_num_classes)
        self.oo_net.layer1 = nn.Sequential(self.oo_net.conv1, self.oo_net.bn1, self.oo_net.relu,
                                            self.oo_net.maxpool, self.oo_net.layer1)
        self.occ_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.occ_fc = nn.Linear(2048, occ_num_classes)

        utils.init_weights(self.do_net, init_type='xavier')
        utils.init_weights(self.oo_net, init_type='xavier')

    def forward(self, img, mask1, mask2):
        """Forward pass.

        Args:
            input data (image, mask1, mask2)

        Returns:
            tensor: disparity, depth order between mask1 and mask2
        """

        layer_1 = self.pretrained.layer1(img) #[1, 256, 96, 96]
        layer_2 = self.pretrained.layer2(layer_1) #[1, 512, 48, 48]
        layer_3 = self.pretrained.layer3(layer_2) #[1, 1024, 24, 24]
        layer_4 = self.pretrained.layer4(layer_3) #[1, 2048, 12, 12]
 
        layer_1_rn = self.scratch.layer1_rn(layer_1) #[1, 256, 96, 96]
        layer_2_rn = self.scratch.layer2_rn(layer_2) #[1, 256, 48, 48]
        layer_3_rn = self.scratch.layer3_rn(layer_3) #[1, 256, 24, 24]
        layer_4_rn = self.scratch.layer4_rn(layer_4) #[1, 256, 12, 12]

        path_4 = self.scratch.refinenet4(layer_4_rn) #[1, 256, 24, 24]
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn) #[1, 256, 48, 48]
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn) #[1, 256, 96, 96]
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn) #[1, 256, 192, 192]
        disp = self.scratch.output_conv(path_1) #[1, 1, 384, 384]

        # predict depth order
        do_feat_1 = self.do_net.layer1(torch.cat([mask1, mask2], dim=1)) #[1, 256, 96, 96]
        do_feat_2 = self.do_net.layer2(do_feat_1 + layer_1)  #[1, 512, 48, 48]
        do_feat_3 = self.do_net.layer3(do_feat_2 + layer_2)  #[1, 1024, 24, 24]
        do_feat_4 = self.do_net.layer4(do_feat_3 + layer_3)  #[1, 2048, 12, 12]

        depth_order = self.depth_avgpool(do_feat_4)
        depth_order = torch.flatten(depth_order, 1)
        depth_order = self.depth_fc(depth_order)

        # predict occlusion order
        oo_feat_1 = self.oo_net.layer1(torch.cat([mask1, mask2], dim=1))
        oo_feat_2 = self.oo_net.layer2(oo_feat_1 + layer_1)
        oo_feat_3 = self.oo_net.layer3(oo_feat_2 + layer_2)
        oo_feat_4 = self.oo_net.layer4(oo_feat_3 + layer_3)

        occ_order = self.occ_avgpool(oo_feat_4)
        occ_order = torch.flatten(occ_order, 1)
        occ_order = self.occ_fc(occ_order)

        return torch.squeeze(disp, dim=1), depth_order, occ_order


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, num_classes=3, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()
        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features,
                                                      use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )
        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: dispairty
        """
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        disp = self.scratch.output_conv(path_1)

        return torch.squeeze(disp, dim=1)
