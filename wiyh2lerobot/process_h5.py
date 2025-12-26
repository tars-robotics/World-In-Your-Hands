import h5py
import os
import numpy as np
import json
from tqdm import tqdm
import cv2
import argparse
from PIL import Image
import multiprocessing as mp
from functools import partial

def process_single_dataset(task):
    """
    处理单个数据集的任务函数
    """
    h5_path, instruction, global_idx = task
    result = {}
    
    try:
        instruction_key = f"data/demo_{global_idx}/instruction"            
        wristlrgb_key = f"data/demo_{global_idx}/obs/eye_in_left_rgb"
        wristrrgb_key = f"data/demo_{global_idx}/obs/eye_in_right_rgb"
        mainrgb_key = f"data/demo_{global_idx}/obs/lf_chest_rgb"
        statel_key = f"data/demo_{global_idx}/left_states"
        stater_key = f"data/demo_{global_idx}/right_states"
        actionl_key = f"data/demo_{global_idx}/left_actions"
        actionr_key = f"data/demo_{global_idx}/right_actions"
        mainpath_key = f"data/demo_{global_idx}/obs/lf_chest_path"
        # print(h5_path)
        with h5py.File(h5_path, "r") as f:
            obs = f['observation'] 
            camera = obs['camera']
            action_left = f['action']['arm_status_feedback']['ldr_camera_pose_in_chest'][()]
            action_right = f['action']['arm_status_feedback']['rdl_camera_pose_in_chest'][()]
            action_index = f['action']['arm_status_feedback']['index'][()]
            action_left_list = []
            action_right_list = []
            state_left_list = []
            state_right_list = []
            
            for i in range(action_index.shape[0]):
                idx = action_index[i]
                if idx + 17 < action_left.shape[0]:
                    left = action_left[idx + 1: idx + 17]
                    right = action_right[idx + 1: idx + 17]
                    state_left = action_left[idx]
                    state_right = action_right[idx]
                    state_left_list.append(state_left)
                    state_right_list.append(state_right)
                    action_left_list.append(left)
                    action_right_list.append(right)
                    
            statel_array = np.array(state_left_list)
            stater_array = np.array(state_right_list)
            actionl_array = np.array(action_left_list)
            actionr_array = np.array(action_right_list)
                
            wristr_image_list = []
            wristl_image_list = []
            main_image_list = []
            main_path_list = []
            act_dir = os.path.dirname(h5_path)
            print(act_dir)
            if not os.path.exists(os.path.join(act_dir, 'camera')):
                return None
            # 处理右侧摄像头图像
            cam_right = camera["rdl_hand_fisheye"]
            filepath = cam_right['filepath'][()]
            index = cam_right['index'][()]
            for i in range(index.shape[0]):
                fp = filepath[i]
                fp_str = fp.decode('utf-8')
                image = Image.open(os.path.join(act_dir, fp_str))
                image = image.resize((640, 512))
                wristr_image_list.append(image)
            wristrrgb = np.array(wristr_image_list)
            
            # # 处理左侧摄像头图像
            cam_left = camera['ldr_hand_fisheye']
            filepath = cam_left['filepath'][()]
            index = cam_left['index'][()]
            for i in range(index.shape[0]):
                fp = filepath[i]
                fp_str = fp.decode('utf-8')
                image = Image.open(os.path.join(act_dir, fp_str))
                image = image.resize((320, 256))
                wristl_image_list.append(image)
            wristlrgb = np.array(wristl_image_list)
            
            cam_main = camera['lf_chest_fisheye']
            filepath = cam_main['filepath'][()]
            index = cam_main['index'][()]
            for i in range(index.shape[0]):
                fp = filepath[i]
                fp_str = fp.decode('utf-8')
                image = Image.open(os.path.join(act_dir, fp_str))
                image = image.resize((320, 256))
                main_image_list.append(image)
                main_path_list.append(os.path.join(act_dir, fp_str))
            mainrgb = np.array(main_image_list)
            mainpath = main_path_list
            # print(mainrgb.shape)
            
            result = {
                'instruction_key': instruction_key,
                'instruction': instruction,
                'wristlrgb_key': wristlrgb_key,
                'wristlrgb_data': wristlrgb,
                'wristrrgb_key': wristrrgb_key,
                'wristrrgb_data': wristrrgb,
                'mainrgb_key': mainrgb_key,
                'mainrgb_data': mainrgb,
                'mainpath_key': mainpath_key,
                'mainpath_data': mainpath,
                'statel_key': statel_key,
                'statel_data': statel_array,
                'stater_key': stater_key,
                'stater_data': stater_array,
                'actionl_key': actionl_key,
                'actionl_data': actionl_array,
                'actionr_key': actionr_key,
                'actionr_data': actionr_array
            }
            
    except Exception as e:
        print(f"Error processing {act_dir}: {e}")
        return None
        
    return result

def collect_all_tasks(data_root):
    """
    收集所有需要处理的任务
    """
    tasks = []
    global_idx = 0
    for scene in sorted(os.listdir(data_root)):
        scene_path = os.path.join(data_root, scene)
        with open(os.path.join(scene_path, f"task.json"), "r") as f:
            task_meta = json.load(f)
        idx = 0
        for action in sorted(os.listdir(scene_path)):
            if not action.startswith("action_"):
                continue
            h5_path = os.path.join(scene_path, action, "dataset.hdf5")
            instruction = task_meta[idx]["task_description_en"]
            tasks.append((h5_path, instruction, global_idx))
            global_idx += 1
            idx += 1
                    
    return tasks

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataroot', type=str, default="data/wiyh", help='data root directory')
    parser.add_argument('--num_workers', type=int, default=1, help='number of worker processes')
    args = parser.parse_args()
    
    data_root = args.dataroot
    # 收集所有任务
    print("Collecting tasks...")
    all_tasks = collect_all_tasks(data_root)
    print(f"Found {len(all_tasks)} tasks to process")
    
    # 创建输出文件
    with h5py.File("data/tmp/wiyh_demo.hdf5", "a") as fw:
        # 使用多进程处理
        print(f"Processing with {args.num_workers} workers...")
        with mp.Pool(processes=args.num_workers) as pool:
            for result in tqdm(pool.imap(process_single_dataset, all_tasks), total=len(all_tasks)):
                if result is not None:
                    fw.create_dataset(result['wristlrgb_key'], data=result['wristlrgb_data'])
                    fw.create_dataset(result['mainpath_key'], data=result['mainpath_data'])
                    fw.create_dataset(result['mainrgb_key'], data=result['mainrgb_data'])
                    fw.create_dataset(result['statel_key'], data=result['statel_data'])
                    fw.create_dataset(result['actionl_key'], data=result['actionl_data'])
                    fw.create_dataset(result['instruction_key'], data=result['instruction'])

if __name__ == "__main__":
    main()