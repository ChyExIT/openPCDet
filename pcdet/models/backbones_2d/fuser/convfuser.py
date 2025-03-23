import torch
from torch import nn

'''
文件实现了一个用于多模态特征融合的卷积融合模块

多模态BEV特征融合
输入特征：
    spatial_features_img：来自图像模态的BEV特征（如Camera数据生成的俯视图特征）
    spatial_features：来自激光雷达模态的BEV特征（如PointPillar处理后的点云特征）


该模块是典型的多模态感知系统中的特征级融合方案，
通过卷积操作实现跨模态特征交互，能有效结合视觉的语义信息与点云的几何信息。

'''

class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1) # 通道维度拼接
        mm_bev = self.conv(cat_bev) # 通过卷积层融合
        batch_dict['spatial_features'] = mm_bev # 输出：融合后的BEV特征将覆盖原有LiDAR特征，供后续检测头使用
        return batch_dict