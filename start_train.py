import torch
import numpy as np
import random
from runner import runner
from train_configs import train_config_maker
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--model_name', default="pointnetpp_dg", type=str, help = "model name. list:   | pointnetpp | dgcnn | pointnetpp_dg")
parser.add_argument('--config_path', default="train_configs/pointnetpp.py", type=str, help = "train config file path.")
parser.add_argument('--experiment_name', default="pointnetpp_dg_test_adv_2", type=str, help = "experiment name.")
parser.add_argument('--input_data_dir_path', default="/home/hiro/3d_tooth_seg/data/data_path", type=str, help = "input data dir path.")
parser.add_argument('--train_data_split_txt_path', default="/home/hiro/3d_tooth_seg/code/fold_path/base_name_train_fold.txt", type=str, help = "train cases list file path.")
parser.add_argument('--val_data_split_txt_path', default="/home/hiro/3d_tooth_seg/code/fold_path/base_name_val_fold.txt", type=str, help = "val cases list file path.")
args = parser.parse_args()

config = train_config_maker.get_train_config(
    args.config_path,
    args.experiment_name,
    args.input_data_dir_path,
    args.train_data_split_txt_path,
    args.val_data_split_txt_path,
)

if args.model_name == "dgcnn":
    from models.dgcnn_model import DGCnnModel
    from models.modules.dgcnn import DGCnnModule
    model = DGCnnModel(config, DGCnnModule)
elif args.model_name == "pointnetpp":
    from models.pointnet_pp_model import PointPpFirstModel
    from models.modules.pointnet_pp import PointPpFirstModule
    model = PointPpFirstModel(config, PointPpFirstModule)
elif args.model_name == "pointnetpp_dg":
    from models.pointnet_pp_dg_model import PointPpFirstModel
    from models.modules.pointnet_pp_dg import PointPpFirstModule
    model = PointPpFirstModel(config, PointPpFirstModule)


runner(config, model)
