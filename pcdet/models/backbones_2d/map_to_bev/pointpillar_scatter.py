import torch
import torch.nn as nn

'''
这个代码文件实现了PointPillars算法中将点云特征映射到鸟瞰图（BEV）的核心散射操作

PointPillarScatter（标准2D版本）
核心功能：
    将点云pillar特征散射到2D BEV网格
    生成伪图像供后续2D卷积网络处理


PointPillarScatter3d（3D改进版）
功能增强：
    支持3D特征散射（保留z轴信息）
    通过特征维度扩展保持空间信息


调用关系
输入依赖：
    需要前置模块生成pillar_features（通常来自PillarVFE模块）
    需要voxel_coords（体素坐标信息）
输出流向：
    生成的spatial_features会传递给2D骨干网络（如BaseBEVBackbone）

    
特性	PointPillarScatter	PointPillarScatter3d
特征维度	纯2D	                3D-aware
内存消耗	较低	                较高
适用场景	常规检测	        需要高度信息的复杂场景
坐标索引计算	(y, x)线性化	(z, y, x)三维线性化


'''

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg # 保存模型配置参数
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # BEV特征维度数
        self.nx, self.ny, self.nz = grid_size # 网格尺寸(nx,ny对应BEV平面分辨率)
        assert self.nz == 1 # 强制要求z轴维度为1（纯2D处理）

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = [] # 存储每个batch的BEV特征
        batch_size = coords[:, 0].max().int().item() + 1  # 从坐标中推断batch大小
        for batch_idx in range(batch_size):
            # 创建全零BEV网格 [C, H*W]
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,  # 展平后的空间维度(nz=1)
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx  # 筛选当前batch的pillars
            this_coords = coords[batch_mask, :]    # 获取对应坐标

            # 计算BEV网格中的一维索引(y * W + x)
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)   # 转为整数索引
            pillars = pillar_features[batch_mask, :]  # 提取当前batch的pillar特征
            pillars = pillars.t()  # 转置为[C, N]格式
            spatial_feature[:, indices] = pillars  # 将特征填充到对应网格位置
            batch_spatial_features.append(spatial_feature)

        # 合并batch维度并reshape为2D特征图  
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            # 初始化3D特征容器 [C/nz, H*W*D]
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict