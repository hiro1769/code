import os
from glob import glob
import numpy as np
import gen_utils as gu
import argparse

# Example class names dictionary, adjust according to your actual labels
class_names_dict = {
    1: "Class 1 Name",
    2: "Class 2 Name",
    3: "Class 3 Name",
    # Add other class mappings here
}

parser = argparse.ArgumentParser(description='Batch Inference Models')
parser.add_argument('--mesh_dir', default="/home/hiro/3d_tooth_seg/data/obj/", type=str, help="Directory containing OBJ files")
parser.add_argument('--gt_json_dir', default="/home/hiro/3d_tooth_seg/data/json/", type=str, help="Directory containing ground truth JSON files")
parser.add_argument('--pred_json_dir', default="/home/hiro/3d_tooth_seg/data/test_results/pointpp_dg_adv_1test", type=str, help="Directory containing predicted JSON files")
args = parser.parse_args()

def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, file_name, is_half=None, vertices=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    iou_list = []
    f1_list = []

    print(f"\nProcessing file: {file_name}")

    for ins_label_name in ins_label_names:
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels == ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        
        # 跳过 class_name 为 0 的情况
        if gt_label_name == 0:
            continue
        
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)

        class_iou = TP / (FP + TP + FN)
        iou_list.append(class_iou)

        # Get the class name from the dictionary
        class_name = class_names_dict.get(gt_label_name, f"Class {gt_label_name}")

        # Calculate Precision and Recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate F1 Score
        class_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(class_f1)

        print(f"{file_name} - {class_name} - IoU: {class_iou:.4f}, F1 Score: {class_f1:.4f}")

        IOU += class_iou
        F1 += class_f1

    if len(iou_list) > 0:
        mIoU = IOU / len(iou_list)
    else:
        mIoU = 0

    if len(f1_list) > 0:
        mF1 = F1 / len(f1_list)
    else:
        mF1 = 0

    print(f"mIoU for {file_name}: {mIoU:.4f}")
    print(f"mF1 for {file_name}: {mF1:.4f}")
    return mIoU, mF1, iou_list, f1_list

mesh_files = sorted(glob(os.path.join(args.mesh_dir, '**/*.obj'), recursive=True))
gt_json_files = sorted(glob(os.path.join(args.gt_json_dir, '**/*.json'), recursive=True))
pred_json_files = sorted(glob(os.path.join(args.pred_json_dir, '**/*.json'), recursive=True))

# Create dictionaries to match files by their base names
mesh_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mesh_files}
gt_json_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_json_files}
pred_json_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_json_files}

# Find common keys
common_keys = set(mesh_dict.keys()) & set(gt_json_dict.keys()) & set(pred_json_dict.keys())

# Ensure that we only process files that exist in all three directories
if len(common_keys) != len(mesh_files) or len(common_keys) != len(gt_json_files) or len(common_keys) != len(pred_json_files):
    print(f"Warning: Some files are missing or extra. Processing {len(common_keys)} matched files.")

# Batch processing
total_IoU = 0
total_F1 = 0
for key in common_keys:
    gt_loaded_json = gu.load_json(gt_json_dict[key])
    gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)

    pred_loaded_json = gu.load_json(pred_json_dict[key])
    pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

    IoU, F1, iou_list, f1_list = cal_metric(gt_labels, pred_labels, pred_labels, key)
    total_IoU += IoU
    total_F1 += F1

average_IoU = total_IoU / len(common_keys)
average_F1 = total_F1 / len(common_keys)

print(f"Average IoU: {average_IoU:.4f}")
print(f"Average F1 Score: {average_F1:.4f}")