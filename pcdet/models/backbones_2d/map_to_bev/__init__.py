from .height_compression import HeightCompression
from .pointpillar_scatter import PointPillarScatter, PointPillarScatter3d
from .conv2d_collapse import Conv2DCollapse

'''
 这个文件实现了3D点云到2D鸟瞰图（BEV）的特征映射组件注册
 核心作用：
 集中管理不同BEV映射策略的注册
 为3D检测框架提供标准化的BEV特征生成接口

HeightCompression	通过高度维度特征压缩生成BEV表示（如取最大高度值）	常规3D物体检测
PointPillarScatter	将点云pillar特征散射到BEV网格（PointPillars标准方法）	PointPillars算法
PointPillarScatter3d	支持3D特征散射的改进版本	需要保留3D信息的改进算法
Conv2DCollapse	使用2D卷积网络压缩特征维度生成BEV表示	需要深度特征提取的场景


采用注册机制动态加载不同映射策略
支持通过配置文件切换BEV生成方式（对应yaml配置中的MAP_TO_BEV参数）


这些组件的具体实现逻辑可以在同目录下的对应文件中查看：

height_compression.py
pointpillar_scatter.py
conv2d_collapse.py
'''


__all__ = {
    'HeightCompression': HeightCompression,
    'PointPillarScatter': PointPillarScatter,
    'Conv2DCollapse': Conv2DCollapse,
    'PointPillarScatter3d': PointPillarScatter3d,
}
