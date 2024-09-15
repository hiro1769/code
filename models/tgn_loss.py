import torch
import sys
sys.path.append("./")
from external_libs.pointnet2_utils.pointnet2_utils import square_distance
DEBUG_NAN = True
def batch_center_offset_loss(pred_offset, sample_xyz, gt_seg_label):
    """offset loss

    Args:
        pred_offset (B, 3, 16000): _description_
        sample_xyz (B, 3, 16000): _description_
        gt_seg_label (B, 1, 16000): _description_
    """
    B, _, N = pred_offset.shape
    
    pred_offset = pred_offset.permute(0,2,1) 
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1) 
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2]) 
    
    centroid_losses = 0
    dir_losses = 0
    centroid_count = 0
    dir_count = 0
    for batch_idx in range(B):
        for tooth_num in range(0, 16):
            cls_cond = gt_seg_label[batch_idx, :] == tooth_num

            cls_sample_xyz = sample_xyz[batch_idx, cls_cond, :]
            if cls_sample_xyz.shape[0] < 5:
                continue
            centroid_count += 1
            cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)

            cls_offset = pred_offset[batch_idx, cls_cond, :]
            cls_offset = cls_offset.view(1, *cls_offset.shape)

            centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
            cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset)
            moved_dists = square_distance(cls_moved_xyz, centroid)
            centroid_losses += torch.div(torch.sum(moved_dists), cls_sample_xyz.shape[1])

            cls_offset_norm = torch.norm(cls_offset, dim=2).view(1,-1,1)
            cls_offset_dir = torch.div(cls_offset, cls_offset_norm)

            points_to_center_dir =  centroid - cls_sample_xyz
            points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
            points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)
            
            cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
            points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]
            if cls_offset_dir.shape[0] != 0:
                dir_count += 1
                dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
                dot_mat = dot_mat - 1
                dot_mat = dot_mat * dot_mat
                
                dir_losses += torch.div(torch.sum(dot_mat), cls_offset_dir.shape[0])
    centroid_losses = torch.div(centroid_losses, centroid_count)
    dir_losses = torch.div(dir_losses, dir_count)
    return centroid_losses, dir_losses

def weighted_batch_center_offset_loss(pred_offset_1, pred_offset_2, sample_xyz, gt_seg_label):
    """offset loss

    Args:
        pred_offset (B, 3, 16000): _description_
        sample_xyz (B, 3, 16000): _description_
        gt_seg_label (B, 1, 16000): _description_
    """
    B, _, N = pred_offset_2.shape
    
    pred_offset_1 = pred_offset_1.permute(0,2,1)
    pred_offset_2 = pred_offset_2.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    
    centroid_losses = 0
    dir_losses = 0
    centroid_count = 0
    dir_count = 0
    for batch_idx in range(B):
        for tooth_num in range(0, 16):
            cls_cond = gt_seg_label[batch_idx, :] == tooth_num

            cls_sample_xyz = sample_xyz[batch_idx, cls_cond, :]
            if cls_sample_xyz.shape[0] < 5:
                continue
            centroid_count += 1
            cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)

            cls_1_offset = pred_offset_1[batch_idx, cls_cond, :]
            cls_1_offset = cls_1_offset.view(1, *cls_1_offset.shape)

            cls_2_offset = pred_offset_2[batch_idx, cls_cond, :]
            cls_2_offset = cls_2_offset.view(1, *cls_2_offset.shape)


            centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)

            cls1_moved_xyz = torch.add(cls_sample_xyz, cls_1_offset)
            moved_dists_1 = torch.sqrt(square_distance(cls1_moved_xyz, centroid)+1e-5)
            weight_1 = moved_dists_1.clone().detach()

            thr=0.1 if tooth_num in [3,4,5,6,7, 11,12,13,14,15] else 0.075
            weight_1[moved_dists_1>=thr] = (weight_1[weight_1>=thr]*10-thr*10)*2 + 1
            weight_1[weight_1>2] = 2
            weight_1[moved_dists_1<thr] = 1

            cls2_moved_xyz = torch.add(cls_sample_xyz, cls_2_offset)
            moved_dists_2 = square_distance(cls2_moved_xyz, centroid)
            centroid_losses += torch.div(torch.sum(moved_dists_2 * weight_1), cls_sample_xyz.shape[1])

            cls_offset_norm = torch.norm(cls_2_offset, dim=2).view(1,-1,1)
            cls_offset_dir = torch.div(cls_2_offset, cls_offset_norm)

            points_to_center_dir =  centroid - cls_sample_xyz
            points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
            points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)

            cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
            points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]
            if cls_offset_dir.shape[0] != 0:
                dir_count = dir_count + 1
                dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
                dot_mat = dot_mat - 1
                dot_mat = dot_mat * dot_mat
                
                dir_losses = dir_losses + torch.div(torch.sum(dot_mat), cls_offset_dir.shape[0])
    if torch.isnan(dir_losses).any() or torch.isnan(centroid_losses).any():
        print(1)
    centroid_losses = torch.div(centroid_losses, centroid_count)
    dir_losses = torch.div(dir_losses, dir_count)
    return centroid_losses, dir_losses

def distance_loss(pred_offset, sample_xyz, gt_seg_label):
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    
    dir_losses = 0
    centroid_losses = 0
    dist_losses = 0
    for i in range(0, 16):
        cls_cond = gt_seg_label==i
        if cls_cond.shape[0]<5:
            continue

        cls_sample_xyz = sample_xyz[cls_cond]
        cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
        cls_offset = pred_offset[cls_cond]
        cls_offset = cls_offset.view(1, *cls_offset.shape)

        centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
        cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset)
        moved_dists = square_distance(cls_moved_xyz, centroid)
        centroid_losses += torch.sum(moved_dists)

        cls_offset_norm = torch.norm(cls_offset, dim=2).view(1,-1,1)
        cls_offset_dir = torch.div(cls_offset, cls_offset_norm)

        points_to_center_dir =  centroid - cls_sample_xyz
        points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
        points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)

        cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
        points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]

            
        if cls_offset_dir.shape[0] != 0:
            dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
            dot_mat = dot_mat - 1
            dot_mat = dot_mat * dot_mat
            
            dir_losses += torch.sum(dot_mat)

    return centroid_losses + dir_losses * 0.1

def distance_loss_with_gin(pred_offset, sample_xyz, gt_seg_label):
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    
    dir_losses = 0
    centroid_losses = 0
    dist_losses = 0
    
    cls_cond = gt_seg_label==-1
    cls_sample_xyz = sample_xyz[cls_cond]
    cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
    cls_offset = pred_offset[cls_cond]
    cls_offset = cls_offset.view(1, *cls_offset.shape)
    centroid_losses += torch.sum(cls_offset**2)*0.01

    for i in range(0, 16):
        cls_cond = gt_seg_label==i
        cls_sample_xyz = sample_xyz[cls_cond]
        cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
        cls_offset = pred_offset[cls_cond]
        cls_offset = cls_offset.view(1, *cls_offset.shape)

        centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
        cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset)
        moved_dists = square_distance(cls_moved_xyz, centroid)
        centroid_losses += torch.sum(moved_dists)

        cls_offset_norm = torch.norm(cls_offset, dim=2).view(1,-1,1)
        cls_offset_dir = torch.div(cls_offset, cls_offset_norm)

        points_to_center_dir =  centroid - cls_sample_xyz
        points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
        points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)

        cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
        points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]

            
        if cls_offset_dir.shape[0] != 0:
            dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
            dot_mat = dot_mat - 1
            dot_mat = dot_mat * dot_mat
            
    return centroid_losses + dir_losses * 0.1

def second_distance_loss(pred_distance, pred_offset, sample_xyz, gt_seg_label):
    pred_distance = pred_distance.permute(0,2,1)
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])

    dir_losses = 0
    centroid_losses = 0
    dist_losses = 0
    for i in range(0, 16):
        cls_cond = gt_seg_label==i
        cls_pred_dist = pred_distance[cls_cond]
        if cls_pred_dist.shape[0]==0:
            continue
        cls_pred_dist = cls_pred_dist.view(1, *cls_pred_dist.shape)
        cls_sample_xyz = sample_xyz[cls_cond]
        cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
        cls_offset = pred_offset[cls_cond]
        cls_offset = cls_offset.view(1, *cls_offset.shape)

        centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
        cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset)
        moved_dists = square_distance(cls_moved_xyz, centroid)
        centroid_losses += torch.sum(moved_dists)
        
        cls_offset_dir = torch.div(cls_offset, torch.norm(cls_offset))
        
        points_to_center_dir =  centroid - cls_sample_xyz
        points_to_center_dir = torch.div(points_to_center_dir, torch.norm(points_to_center_dir))

        dir_losses += -(torch.dot(cls_offset_dir.view(-1), points_to_center_dir.view(-1)))

    return centroid_losses + dir_losses

def batch_chamfer_distance_loss(pred_offset, sample_xyz, gt_seg_label):
    """offset loss

    Args:
        pred_offset (B, 3, 16000): _description_
        sample_xyz (B, 3, 16000): _description_
        gt_seg_label (B, 1, 16000): _description_
    """
    B, _, N = pred_offset.shape
    
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    
    centroids = []
    for batch_idx in range(B):
        b_centroids = []
        for tooth_num in range(0, 16):
            cls_cond = gt_seg_label[batch_idx, :] == tooth_num 

            cls_sample_xyz = sample_xyz[batch_idx, cls_cond, :] 
            if cls_sample_xyz.shape[0] < 5:
                continue
            centroid = torch.mean(cls_sample_xyz, dim=0).view(3)
            b_centroids.append(centroid)
        b_centroids = torch.stack(b_centroids)
        centroids.append(b_centroids)
    loss = 0
    for batch_idx in range(B):
        moved_points = sample_xyz[batch_idx, :] + pred_offset[batch_idx, :]
        moved_points = moved_points[gt_seg_label[batch_idx]!=-1, :]
        b_centroids = centroids[batch_idx]
        pred_ct_dists = square_distance(moved_points.unsqueeze(dim=0), b_centroids.unsqueeze(dim=0))
        sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
        min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]
        ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
        loss += torch.sum(ratio)/moved_points.shape[0]
    loss /= B
    return loss

def chamfer_distance_with_gin_loss(pred_offset, sample_xyz, centroid):
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(sample_xyz, pred_offset)

    pred_ct_dists = square_distance(pred_centroid, centroid)
    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]

    ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
    
    loss = torch.sum(ratio)
    return loss

def compute_stat(label, current_xyz, gt):
    """计算标签的统计信息（均值和方差），处理除零错误"""
    B, _, N = label.shape
    label = label.view(1, -1)
    if gt == True:
        label = label + 1
    batch_size = label.shape[0]  # 获取 batch 大小
    num_classes = 16  # 假设有 16 个类
    num_features = 3  # 每个点有 3 个坐标 (x, y, z)
    stats = torch.zeros((batch_size, num_classes, 2 * num_features), device=current_xyz.device)  # 最终的统计量 [batch_size, 16, 6]

    epsilon = 1e-6  # 防止方差为 0 的保护值

    for batch_idx in range(batch_size):
        # 对每个 batch 处理
        seg_label = label[batch_idx]  # 当前 batch 的分割标签, shape [24000]
        xyz = current_xyz[batch_idx]  # 当前 batch 的点云坐标, shape [3, 24000]

        for class_id in range(1, num_classes + 1):  # 类别 ID 从 1 到 16
            # 获取当前类的掩码
            mask = (seg_label == class_id)  # 掩码 shape [24000]

            if mask.sum() > 0:  # 如果该类有点
                # 获取该类所有选中的点
                mask_indices = torch.nonzero(mask, as_tuple=True)[0]  # 获取掩码中为 True 的索引
                points = xyz[:, mask_indices]  # 筛选出属于该类的点, points shape [3, num_selected_points]

                # 计算每个维度的均值和方差
                mean = points.mean(dim=1)  # 按列取均值，返回 [3]
                variance = points.var(dim=1, unbiased=False)  # 使用无偏方差，返回 [3]
                variance = torch.clamp(variance, min=epsilon)  # 防止方差为0，避免出现nan

                # 将均值和方差存储到 stats 张量中
                stats[batch_idx, class_id - 1, :num_features] = mean  # 将均值存入 stats
                stats[batch_idx, class_id - 1, num_features:] = variance  # 将方差存入 stats
            else:
                stats[batch_idx, class_id - 1, :] = torch.zeros(6, device=current_xyz.device)  # 填充全零向量

    return stats

def compute_centroids(labels, points, gt):
    """
    计算每个类别的质心。

    Args:
        labels (Tensor): 标签，形状为 [B, 1, N]
        points (Tensor): 点云坐标，形状为 [B, 3, N]
        gt (bool): 是否为 ground truth

    Returns:
        Tensor: 质心信息，形状为 [B, 16, 6]（均值和方差）
    """
    B, _, N = labels.shape
    labels = labels.view(B, -1)
    if gt:
        labels = labels + 1  # 将标签范围从 [-1, 15] 转为 [0, 16]
    
    batch_size = labels.shape[0]
    num_classes = 16
    num_features = 3
    stats = torch.zeros((batch_size, num_classes, 2 * num_features), device=points.device)
    
    epsilon = 1e-6  # 防止方差为0

    
    # 转置 points 为 [B, N, 3]
    points = points.permute(0, 2, 1)  # 从 [B, 3, N] 转为 [B, N, 3]
    
    for batch_idx in range(batch_size):
        for class_id in range(1, num_classes + 1):
            mask = (labels[batch_idx] == class_id)  # 形状为 [N]
            
            if mask.sum() > 0:
                selected_points = points[batch_idx][mask]  # 形状为 [M, 3]
                
                # 计算均值和方差
                mean = selected_points.mean(dim=0)  # 形状为 [3]
                variance = selected_points.var(dim=0, unbiased=False)  # 形状为 [3]
                variance = torch.clamp(variance, min=epsilon)  # 防止方差为0
                
                # 存储均值和方差
                stats[batch_idx, class_id - 1, :num_features] = mean
                stats[batch_idx, class_id - 1, num_features:] = variance
            else:
                # 如果没有该类别的点，填充为零
                stats[batch_idx, class_id - 1, :] = torch.zeros(2 * num_features, device=points.device)
    
    return stats


import torch.nn.functional as F

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        target = F.one_hot(target, num_classes=10)
        x = x.permute(0,2,1)
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        #target = F.one_hot(target, num_classes=10)
        #pred = pred.permute(0,2,1)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def tooth_class_loss(cls_pred, gt_cls, cls_num, weight=None, label_smoothing=None):
    """
    Input
        cls_pred: [B, C, N] (e.g., [1, 17, 16000])
        gt_cls: [B, 1, N] (e.g., [1, 1, 16000]) -> -1 is background, 0~15 is foreground
    """
    B, _, N = gt_cls.shape
    gt_cls = gt_cls.view(B, -1).long()  # 将 gt_cls 转换为 [B, N] 并指定为 long 类型
    gt_cls = gt_cls + 1  # -1~16 -> 0~17
    
    if label_smoothing is None:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).float().cuda() if weight is not None else None)
        loss = criterion(cls_pred, gt_cls)
    else:
        loss = LabelSmoothingLoss(cls_num, smoothing=label_smoothing)(cls_pred, gt_cls)
    
    return loss

def uncert_tooth_class_loss(cls_pred_1, cls_pred_2, gt_cls,weight):
    """
    Input
        cls_pred_1: 1, 17, 16000
        gt_cls: 1, 1, 16000 -> -1 is background, 0~15 is foreground
    """
    B, _, N = gt_cls.shape
    gt_cls = gt_cls.view(B, -1)
    gt_cls = gt_cls.type(torch.long)
    gt_cls = gt_cls + 1
    loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(cls_pred_1, gt_cls)
    




def make_teeth_mask(gt_seg_label):
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[0],-1)
    gt_bin_label = torch.ones_like(gt_seg_label)
    gt_bin_label[gt_seg_label == -1] = 0
    return gt_bin_label

def make_teeth_mask_binary(gt_seg_label):
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[0],-1)
    gt_bin_label = torch.ones_like(gt_seg_label)
    gt_bin_label[gt_seg_label == -1] = 0
    gt_bin_label = gt_bin_label.type(torch.long)
    gt_bin_label = gt_bin_label.view(gt_seg_label.shape[0], -1)
    return gt_bin_label

def teeth_mask_loss(pred_mask, gt_seg_label):
    gt_bin_label = make_teeth_mask(gt_seg_label).type(torch.float32).reshape(pred_mask.shape)
    return torch.nn.BCEWithLogitsLoss()(pred_mask, gt_bin_label)
    
def weighted_cls1_loss(pred_weight_1, pred_cls_1, gt_seg_label, weight):
    B, _, N = gt_seg_label.shape
    gt_seg_label = gt_seg_label.view(B, -1)
    gt_seg_label = gt_seg_label.type(torch.long)
    gt_seg_label = gt_seg_label + 1
    bce_1 = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")(pred_cls_1, gt_seg_label)

    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    #pred_weight_1: B, 16000
    loss = (bce_1 * pred_weight_1) ** 2 + (1-pred_weight_1)**2
    loss = torch.sum(loss)/loss.shape[1]
    return loss

def weighted_cls2_loss(pred_weight_1, pred_cls_2, gt_seg_label, weight):
    B, _, N = gt_seg_label.shape
    gt_seg_label = gt_seg_label.view(B, -1)
    gt_seg_label = gt_seg_label.type(torch.long)
    gt_seg_label = gt_seg_label + 1
    bce_2 = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")(pred_cls_2, gt_seg_label)

    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    loss = (2.0-pred_weight_1)*bce_2
    loss = torch.sum(loss)/loss.shape[1]
    return loss