import argparse
import os
import numpy as np
from glob import glob
import gen_utils as gu

parser = argparse.ArgumentParser()
parser.add_argument('--source_obj_data_path', default="/home/hiro/3d_tooth_seg/data/obj", type=str, help="data path in which original .obj data are saved")
parser.add_argument('--source_json_data_path', default="/home/hiro/3d_tooth_seg/data/json", type=str, help="data path in which original .json data are saved")
parser.add_argument('--save_data_path', default="/home/hiro/3d_tooth_seg/data/data_path_33", type=str, help="data path in which processed data will be saved")
args = parser.parse_args()

SAVE_PATH = args.save_data_path
SOURCE_OBJ_PATH = args.source_obj_data_path
SOURCE_JSON_PATH = args.source_json_data_path
Y_AXIS_MAX = 33.15232091532151
Y_AXIS_MIN = -36.9843781139949

os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

stl_path_ls = []
"""
stl_path_ls: obj文件路径的列表
"""
for dir_path in [
    x[0] for x in os.walk(SOURCE_OBJ_PATH)
    ][1:]:
    stl_path_ls += glob(os.path.join(dir_path,"*.obj"))#['/home/repos/ToothGroupNetwork/data/3D_scans_per_patient_obj_files/01FUXUMF/01FUXUMF_upper.obj', '/home/repos/ToothGroupNetwork/data/3D_scans_per_patient_obj_files/01FUXUMF/01FUXUMF_lower.obj']

json_path_map = {}
"""
json_path_map: json文件的路径字典, 键为文件名(不含扩展名), 值为文件的完整路径
"""
for dir_path in [
    x[0] for x in os.walk(SOURCE_JSON_PATH)
    ][1:]:
    for json_path in glob(os.path.join(dir_path,"*.json")):
        json_path_map[os.path.basename(json_path).split(".")[0]] = json_path #将文件名 (不含扩展名) 作为键，文件的完整路径作为值，添加到 json_path_map 字典中。

all_labels = []
for i in range(len(stl_path_ls)):
    print(i, end=" ")
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0] #obj文件名不含扩展名，和json文件名相同
    loaded_json = gu.load_json(json_path_map[base_name]) #加载json文件
    labels = np.array(loaded_json['labels']).reshape(-1,1) 
    """
    labels: 将json文件中的labels字段转换为numpy数组,一维列数组
    
    """
    # # 如果是下颚，先将标签减去20
    # if loaded_json['jaw'] == 'lower':
    #     labels -= 20  # 将下颚的标签范围从31~48变为11~28

    #     # 处理下颚标签
    #     labels[labels // 10 == 1] = (labels[labels // 10 == 1] % 10) + 16  # 将标签11~18转换为17~24
    #     labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 24  # 将标签21~28转换为25~32
    # else:
    #     # 处理上颚标签
    #     labels[labels // 10 == 1] %= 10  # 将标签11~18转换为1~8
    #     labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8  # 将标签21~28转换为9~16

    # # 将背景标签转化为0
    # labels[labels <= 0] = 0  # 将所有小于等于0的标签设置为背景（0）

    
    if loaded_json['jaw'] == 'lower':
        labels -= 20
    labels[labels//10==1] %= 10
    labels[labels//10==2] = (labels[labels//10==2]%10) + 8
    labels[labels<0] = 0

      
    vertices, org_mesh = gu.read_txt_obj_ls(stl_path_ls[i], ret_mesh=True, use_tri_mesh=False) #ret_mesh=True返回三角网格，use_tri_mesh=False不使用trimesh库加载obj
    
    """
    vertices: 顶点坐标和法线向量拼接后的 NumPy 数组，形状为 (顶点数, 6)
    org_mesh: 存储的原始的网格对象，包含了 OBJ 文件中的顶点、面和法线信息
    """

    vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)#所有顶点的坐标中心化
    #vertices[:, :3] = ((vertices[:, :3]-vertices[:, 1].min())/(vertices[:, 1].max() - vertices[:, 1].min()))*2-1
    vertices[:, :3] = ((vertices[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX - Y_AXIS_MIN))*2-1 #将所有顶点的y坐标归一化到[-1,1]之间

    labeled_vertices = np.concatenate([vertices,labels], axis=1) 
    #labeled_vertices: 将顶点坐标,法向量和标签拼接在一起[N,7]

    name_id = str(base_name)
    if labeled_vertices.shape[0]>24000:
        labeled_vertices = gu.resample_pcd([labeled_vertices], 24000, "fps")[0]#fps: 最远点取样

    np.save(os.path.join(SAVE_PATH, f"{name_id}_sampled_points"), labeled_vertices)
