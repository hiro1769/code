import torch.nn as nn
import torch.nn.functional as F
from external_libs.pointnet2_utils.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
import torch
from .dgcnn import get_graph_feature
class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.cls_pred = True
        input_feature_num=6
        scale = 1
        scale_1 = 4
        self.k = 25
        self.scale = 1
        drop_out_ratio = 0.5
        #目标点的数量、球查询半径、球内的最大采样数量、输入特征数量（位置加上特征向量）、MLP的数量、group_all为False
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.025, 0.05], [32, 64], input_feature_num, [[16*scale, 32*scale], [32*scale, 32*scale]])#两次MLp和两个半径
        self.sa2 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [32, 64], 64*scale, [[64*scale, 128*scale], [64*scale, 128*scale]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 256*scale, [[256*scale, 512*scale], [256*scale, 512*scale]])
        

        self.fp3 = PointNetFeaturePropagation((512+64)*scale_1, [256*scale_1, 256*scale_1]) #in_channel, [mlp1, mlp2]
        self.fp2 = PointNetFeaturePropagation((256+16)*scale_1, [128*scale_1, 128*scale_1])
        self.fp1 = PointNetFeaturePropagation((128*scale_1)+input_feature_num, [64*scale_1, 32*scale_1])

        self.offset_conv_1 = nn.Conv1d(32*scale_1,16, 1)
        self.offset_bn_1 = nn.BatchNorm1d(16)
        self.dist_conv_1 = nn.Conv1d(32*scale_1,16, 1)
        self.dist_bn_1 = nn.BatchNorm1d(16)
        
        self.offset_conv_2 = nn.Conv1d(16,3, 1)
        self.dist_conv_2 = nn.Conv1d(16,1, 1)

        self.bn3 = nn.BatchNorm2d(64*self.scale)
        self.bn4 = nn.BatchNorm2d(64*self.scale)

        self.conv3 = nn.Sequential(nn.Conv2d(128*self.scale, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv4 = nn.Sequential(nn.Conv2d(64*self.scale, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn5 = nn.BatchNorm2d(256*self.scale)
        self.bn6 = nn.BatchNorm2d(256*self.scale)
        
        self.conv5 = nn.Sequential(nn.Conv2d(512*self.scale, 256*self.scale, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256*self.scale, 256*self.scale, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn7 = nn.BatchNorm2d(2048*self.scale)
        
        self.conv7 = nn.Sequential(nn.Conv2d(2048*self.scale, 2048*self.scale, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=drop_out_ratio)

        if self.cls_pred:
            self.cls_conv_1 = nn.Conv1d(32*scale_1,17, 1)
            self.cls_bn_1 = nn.BatchNorm1d(17)
            self.cls_conv_2 = nn.Conv1d(17,17, 1)

        nn.init.zeros_(self.offset_conv_2.weight)
        nn.init.zeros_(self.dist_conv_2.weight)

        #prediction part
        self.conv1 = nn.Conv1d(32, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        
    #输入批次、通道（包括xyz和其他特征）、批次中的采样数据。
    def forward(self, xyz_in):#xyz_in [[B,6,N],[B,1,N]]
        xyz = xyz_in[0]
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        #sa1
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)#[b,3,1024],[b,64,1024]
        #ec1
        l1_points = get_graph_feature(l1_points, k=self.k)    
        l1_points = self.conv3(l1_points)
        l1_points = self.conv4(l1_points)                      
        l1_points = l1_points.max(dim=-1, keepdim=False)[0]#[b,128,1024]    
        #sa2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)#[b,3,512],[b,256,512]
        #ec2
        l2_points = get_graph_feature(l2_points, k=self.k)    
        l2_points = self.conv5(l2_points)
        l2_points = self.conv6(l2_points)                       
        l2_points = l2_points.max(dim=-1, keepdim=False)[0]#[b,512,512]    
        #sa3
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)#[b,3,256],[b,1024,256]
        #ec3
        l3_points = get_graph_feature(l3_points, k=self.k)    
        l3_points = self.conv7(l3_points)                       
        l3_points = l3_points.max(dim=-1, keepdim=False)[0]#[b,2048,256]  
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)#[b,1024,512]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)#[b,512,1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)#[b,128,24000]

        #x = F.relu(self.bn1(self.conv1(l0_points)))
        
        offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(l0_points)))
        offset_result = self.offset_conv_2(offset_result)

        dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(l0_points)))
        dist_result = self.dist_conv_2(dist_result)

        output = [l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result]
        
        if self.cls_pred:
            cls_pred = F.relu(self.cls_bn_1(self.cls_conv_1(l0_points)))
            # cls_pred = self.dp1(cls_pred)
            cls_pred = self.cls_conv_2(cls_pred)
            output.append(cls_pred)

        return output


class PointPpFirstModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.first_sem_model = get_model()

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, cls_pred = self.first_sem_model(inputs)
        outputs = {
            "cls_pred": cls_pred,
            "l0_xyz": l0_xyz,
            "l0_points": l0_points,
        }
        return outputs

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import torch
    model = get_model()
    xyz = torch.rand(6, 6, 2048)
    #output is B, C, N order
    for item in model(xyz):
        print(item.shape)
    input()