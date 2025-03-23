
# 继承自Detector3DTemplate基类，属于单阶段3D检测器
from .detector3d_template import Detector3DTemplate

# 专为处理点云数据设计（特别是自动驾驶场景的LiDAR数据）
class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        '''
        Args:
            model_cfg: 模型配置参数
            num_class: 目标类别数
            dataset: 数据集类对象
        
        Returns:
            None
        
        Notes:
            1. 通过build_networks()创建处理流水线（具体实现应在父类/其他模块）
            2. 典型包含：
                Pillar特征编码层（将点云转换为伪图像）
                2D卷积骨干网络
                检测头（RPN）
                点云特征提取头（FPN）
        '''
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks() 

    def forward(self, batch_dict):
        '''
        Args:
            batch_dict: 数据字典（包含点云数据及其特征）

        Returns:
            ret_dict: 返回字典
            tb_dict: TensorBoard字典
            disp_dict: 显示字典
            训练：损失值及监控指标
            推理：预测框及召回率指标
        '''
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:  # 训练时：计算RPN损失（通过get_training_loss()）
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict # 返回损失值、TensorBoard字典、显示字典
        else:  # 推理时：后处理（通过post_processing()）
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts # 返回预测结果、召回字典

    def get_training_loss(self):
        disp_dict = {}

        # 分类损失（通常为focal loss）
        # 回归损失（通常为smooth L1 loss）
        loss_rpn, tb_dict = self.dense_head.get_loss() # 检测头计算损失
        tb_dict = { # TensorBoard字典
            'loss_rpn': loss_rpn.item(), # RPN损失
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict # 返回RPN损失、TensorBoard字典、显示字典
