import torch
from loss_meter import LossMap
from models.base_model import BaseModel
from . import tgn_loss

class PointPpFirstModel(BaseModel):
    def __init__(self, config, module):
        super(PointPpFirstModel, self).__init__(config, module)
        from models.modules.pointnet_pp_dg import Discriminator
        self.discriminator = Discriminator().cuda()
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.adversarial_loss = torch.nn.BCELoss()
        # 初始化自适应权重 λ1 和 λ2
        self.lambda_1 = torch.nn.Parameter(torch.ones(1, device='cuda'), requires_grad=True)
        self.lambda_2 = torch.nn.Parameter(torch.ones(1, device='cuda'), requires_grad=True)
        
        # 根据配置文件初始化优化器
        optimizer_config = config["tr_set"]["optimizer"]
        if optimizer_config["NAME"] == 'adam':
            self.optimizer = torch.optim.Adam(
                list(self.module.parameters()) + list(self.discriminator.parameters()) + [self.lambda_1, self.lambda_2],
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"]
            )
        elif optimizer_config["NAME"] == 'sgd':
            self.optimizer = torch.optim.SGD(
                list(self.module.parameters()) + list(self.discriminator.parameters()) + [self.lambda_1, self.lambda_2],
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"],
                momentum=0.9
            )
    
        
    def get_loss(self, gt_seg_label_1, sem_1, xyz, point): 
        # 计算分类损失
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, gt_seg_label_1, 17)
        sem_1 = torch.argmax(sem_1, dim=1)
        point = point[:, :3, :]
        
        # 计算统计量
        real_stat = tgn_loss.compute_stat(gt_seg_label_1, xyz)
        fake_stat = tgn_loss.compute_stat(sem_1, point)
        
        # 判别器输出
        real_output = self.discriminator(real_stat)
        fake_output = self.discriminator(fake_stat)
        
        # 计算对抗损失
        real_loss = self.adversarial_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.adversarial_loss(fake_output, torch.zeros_like(fake_output))
        loss_D = real_loss + fake_loss

        return {
            "ce_loss": (tooth_class_loss_1, self.lambda_1.item()),  # 分类损失
            "adv_loss": (loss_D, self.lambda_2.item())  # 对抗损失
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]
        
        if phase == "train":
            output = self.module(inputs)
        else:
            with torch.no_grad():
                output = self.module(inputs)
        
        # 初始化 loss_meter
        loss_meter = LossMap()

        # 获取损失
        loss_dict = self.get_loss(seg_label, output["cls_pred"], output["l0_xyz"], output["l0_points"])
        loss_meter.add_loss_by_dict(loss_dict)

        if phase == "train":
            # 自适应损失计算
            tooth_class_loss_1 = loss_dict["ce_loss"][0]
            loss_D = loss_dict["adv_loss"][0]

            # 自适应总损失 = 分类损失 / λ1^2 + 对抗损失 / λ2^2 + 正则化项
            total_loss = (1 / self.lambda_1**2) * tooth_class_loss_1 + (1 / self.lambda_2**2) * loss_D
            regularization = self.lambda_1**2 + self.lambda_2**2
            print("self.lambda_1:", self.lambda_1.item(), "self.lambda_2:", self.lambda_2.item())
            total_loss = total_loss + regularization
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer.step()
            
        return loss_meter

    def infer(self, batch_idx, batch_item, **options):
        # 根据需要实现推理逻辑
        pass