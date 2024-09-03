import torch
from . import tgn_loss
from models.base_model import BaseModel
from loss_meter import LossMap

class PointPpFirstModel(BaseModel):
    def get_loss(self, gt_seg_label_1, sem_1, xyz, point): 
        # 计算类别损失
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, gt_seg_label_1, 17)
        
        # 拼接真实坐标和真实标签
        gt_xyz = torch.cat((xyz, gt_seg_label_1), dim=1)
        
        # 拼接预测坐标和预测标签
        sem_1 = torch.argmax(sem_1, dim=1).unsqueeze(1).float()  # 将预测标签进行 argmax 操作
        pre_xyz = torch.cat((point[:, :3, :], sem_1), dim=1)
        
        centroid_dist = []
        
        # 获取唯一的标签列表，排除背景标签 -1
        unique_labels = torch.unique(gt_seg_label_1)
        unique_labels = unique_labels[unique_labels >= 0]
        
        for label in unique_labels:
            if label == -1:
                continue  # 跳过标签为-1的点
            
            # 生成掩码并调整形状以匹配xyz
            mask = gt_seg_label_1.squeeze(1) == label  # 掩码形状为 [1, 24000]
            mask = mask.unsqueeze(1).expand_as(xyz)  # 调整掩码形状为 [1, 3, 24000]
            
            # 获取真实标签为当前label的点的坐标
            gt_label_pts = xyz[mask].view(3, -1).t()  # 提取相应标签的真实坐标，形状为 [num_pts, 3]
            
            if gt_label_pts.size(0) == 0:
                continue  # 如果没有对应标签的点，则跳过
            
            # 计算真实质心
            gt_centroid = torch.mean(gt_label_pts, dim=0)  # 计算质心
            
            # 生成预测掩码并调整形状以匹配pre_xyz
            mask_pre = sem_1.squeeze(1) == label  # 掩码形状为 [1, 24000]
            mask_pre = mask_pre.unsqueeze(1).expand_as(point[:, :3, :])  # 调整掩码形状为 [1, 3, 24000]
            
            # 获取预测标签为当前label的点的坐标
            pre_label_pts = point[:, :3, :][mask_pre].view(3, -1).t()  # 提取相应标签的预测坐标，形状为 [num_pts, 3]
            
            if pre_label_pts.size(0) == 0:
                continue  # 如果没有对应标签的点，则跳过
            
            # 计算预测质心
            pre_centroid = torch.mean(pre_label_pts, dim=0)  # 计算质心
            
            # 计算质心之间的欧氏距离并保存
            centroid_dist.append(torch.sqrt(torch.sum(torch.square(pre_centroid - gt_centroid))))
        
        # 计算质心损失
        if len(centroid_dist) > 0:
            centroid_loss = sum(centroid_dist) / len(centroid_dist)
        else:
            centroid_loss = torch.tensor(0.0, device=gt_seg_label_1.device)
        
        # # 返回总损失
        # total_loss = tooth_class_loss_1 + centroid_loss
        # return total_loss

        return {
            "ce_loss": (tooth_class_loss_1, 1), "centroid_loss": (centroid_loss, 0)
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        
        #centroids = batch_item[1].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]
        
        if phase == "train":
            output = self.module(inputs)
        else:
            with torch.no_grad():
                output = self.module(inputs)
        loss_meter = LossMap()
        
        loss_meter.add_loss_by_dict(self.get_loss(
            seg_label, 
            output["cls_pred"], 
            output["l0_xyz"],
            output["l0_points"]
            )
        )
        
        if phase == "train":
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad() #梯度清零
            loss_sum.backward() #反向传播
            self.optimizer.step()

        return loss_meter

    def infer(self, batch_idx, batch_item, **options):
        pass