import os
import json
import numpy as np
import h5py
import sys
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Iterable, Optional, Dict
import laspy
from scipy.spatial.transform import Rotation as R

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("wiyh dev-kit only supports Python version 3.")
MAX_DEPTH=2

class WIYH:
    
    def __init__(self,
                 version: str = 'v1.0-train',
                 dataroot: str = '/data/wiyh',
                 trajectory_range: int = 50
                 ):
        self.version = version
        self.dataroot = dataroot
        self.left_tracker_id = "left_hand"
        self.right_tracker_id = "right_hand"
        self.chest_tracker_id = "chest"
        self.trajectory_range = trajectory_range
        
        
    def vis_h5_structure(self, scene, eps, output_txt=None):
        """
        print and save the HDF5 file structure as a tree view.
        """ 
        def build_tree(item, prefix='', is_last=True, lines=None):
            if lines is None:
                lines = []
            
            name = item.name.split('/')[-1] or '/'
            
            if item.name == '/': 
                branch = ""
                current_line = "📁 /"
            else:
                branch = "└── " if is_last else "├── "
                
                if isinstance(item, h5py.Group):
                    current_line = f"{branch}📁 {name}/"
                else:
                    current_line = f"{branch}📊 {name} (shape: {item.shape}, dtype: {item.dtype})"
            
            lines.append(f"{prefix}{current_line}")
            
            if isinstance(item, h5py.Group):
                items = list(item.items())
                for i, (key, sub_item) in enumerate(items):
                    is_last_child = (i == len(items) - 1)
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    build_tree(sub_item, new_prefix, is_last_child, lines)
            
            return lines
        
        try:
            file_path = os.path.join(self.dataroot, scene, eps, "dataset.hdf5")
            with h5py.File(file_path, 'r') as f:
                tree_lines = build_tree(f)
                
                # compile output content
                output_content = [
                    f"HDF5 Tree Structure",
                    f"File path: {os.path.abspath(file_path)}",
                    f"Analysis time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 50,
                    ""
                ]
                output_content.extend(tree_lines)
                # write to console
                print('\n'.join(output_content))
                if output_txt is not None:
                    
                    with open(output_txt, 'w', encoding='utf-8') as txt_file:
                        txt_file.write('\n'.join(output_content))         
                    print(f"\nOutput tree structure to: {output_txt}")
                
                return output_txt
                
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def list_scenes(self):
        scene_list = sorted(os.listdir(self.dataroot))
        print(f"{len(scene_list)} Scenes in the dataset: \n")
        print("Scene Names:\n")
        
        for scene in scene_list:
            print(f"{scene}\n")
        return scene_list
    
    def get_scene_meta(self, scene):
        try:
            meta_path = os.path.join(self.dataroot, scene, "task.json")
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                print(f"Metadata for scene {scene}:\n")
                print(json.dumps(meta, indent=4))
                return meta
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def list_eps(self, scene):
        eps_list = sorted(os.listdir(os.path.join(self.dataroot, scene)))
        eps_list = [i for i in eps_list if i.startswith('action_')]
        print(f"{len(eps_list)} Episodes in the scene {scene}: \n")
        print("Episode Names:\n")
        
        for eps in eps_list:
            print(f"{eps}\n")
        return eps_list
    
    def get_eps_len(self, scene, eps):
        try:
            file_path = os.path.join(self.dataroot, scene, eps, "dataset.hdf5")
            with h5py.File(file_path, 'r') as f:
                frames = f['observation']['camera']['ldl_hand_fisheye']['index'][()]
                length = frames.shape[0]
                print(f"Episode {eps} in scene {scene} has {length} frames.")
                return length
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def get_eps_meta(self, scene, eps):
        eps_id = int(eps.split('_')[-1])
        try:
            meta_path = os.path.join(self.dataroot, scene, "task.json")
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)[eps_id]
                print(f"Metadata for episode {eps} in scene {scene}:\n")
                print(json.dumps(meta, indent=4))
                return meta
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_annoatations(self, scene, eps, anno_type):
        
        if anno_type not in ['cot', 'instruct', 'task_status']:
            raise ValueError("anno_type must be one of ['cot', 'instruct', 'task_status']")
        try:
            h5_path = os.path.join(self.dataroot, scene, eps, "dataset.hdf5")
            with h5py.File(h5_path, 'r') as f:
                if anno_type == 'instruct':
                    annotations = f['annotation']['atomic_task_description']['atomic_task_description'][()]
                    annotations = [ann.decode('utf-8') for ann in annotations]
                    print(f"Instruction Annotations for episode {eps} in scene {scene}:\n")
                    print(annotations[0])
                    return annotations[0]
                elif anno_type == 'task_status':
                    annotations = f['annotation']['atomic_task_status']['atomic_task_status'][()]
                    annotations = [ann.decode('utf-8') for ann in annotations]
                    print(f"Task Status Annotations for episode {eps} in scene {scene}:\n")
                    print(annotations)
                    return annotations
                elif anno_type == 'cot':
                    annotations = f['meta']['cot'][()]
                    annotations = [ann.decode('utf-8') for ann in annotations]
                    print(f"COT Annotations for episode {eps} in scene {scene}:\n")
                    for ann in annotations:
                        print(ann)
                    return annotations
        except Exception as e:
            print(f"Error: {e}")
            return None

    def show_all_camera_rgb(self, scene: str, action: str, camera_list: list, index: int):
        """
        Load and display RGB images from multiple cameras in a grid layout.

        Parameters:
            worldcode (str): Unique identifier for the dataset/world code
            action (str): Action label (e.g., "action_000") for the target frame
            camera_list (list): List of camera names to visualize

        Returns:
            None: Displays a matplotlib figure with all camera images
        """
        # Calculate grid layout (fixed 2 rows x 3 columns for 6 cameras)
        n_cameras = len(camera_list)
        n_rows = 2
        n_cols = 3

        # Create figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()  # Flatten 2D axes array to 1D for easy iteration

        # Iterate through each camera to load and plot images
        for idx, camera in enumerate(camera_list):
            try:
                # Load RGB image for the current camera
                filepath, img = self.visualize_rgb(scene, action, camera, index)

                # Plot image and add camera name as title
                axes[idx].imshow(img)
                axes[idx].set_title(camera, fontsize=12, fontweight='bold')
                axes[idx].axis('off')  # Hide axis ticks/labels

            except Exception as e:
                # Handle image loading failures gracefully
                axes[idx].text(0.5, 0.5, f"Load failed:\n{camera}", 
                            ha='center', va='center', fontsize=10, color='red')
                axes[idx].axis('off')
                print(f"Warning: Failed to load camera {camera}. Error: {str(e)}")

        # Hide unused subplots if camera count < 6
        for idx in range(n_cameras, n_rows * n_cols):
            axes[idx].axis('off')

        # Adjust layout to prevent title/image overlap
        plt.tight_layout()
        # Display the final figure
        plt.show()    
    
    def visualize_rgb(self, scene, eps, camera, index):  
        try:
            h5_path = os.path.join(self.dataroot, scene, eps, "dataset.hdf5")
            with h5py.File(h5_path, 'r') as f:
                if camera != 'depth':
                    filepath = f['observation']['camera'][camera]['filepath'][()][index].decode('utf-8')
                    filepath = os.path.join(self.dataroot, scene, eps, filepath)
                    img = cv2.imread(filepath)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(f"Visualizing {camera} for episode {eps} in scene {scene}")
                    return filepath, img
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def visualize_depth(self, scene, eps, index):
        try:
            h5_path = os.path.join(self.dataroot, scene, eps, "dataset.hdf5")
            print(f"Visualizing depth for episode {eps} in scene {scene}")
            with h5py.File(h5_path, 'r') as f:
                depthpath = f['observation']['pointcloud']['chest']['filepath'][()][index]
                depthpath = os.path.join(self.dataroot, scene, eps, depthpath.decode('utf-8'))
                las = laspy.read(str(depthpath))
                pointcloud = np.vstack((las.x, las.y, las.z)).T
                
                rgbpath = f['observation']['camera']['lf_chest_fisheye']['filepath'][()][index]
                rgbpath = os.path.join(self.dataroot, scene, eps, rgbpath.decode('utf-8'))
                image = cv2.imread(rgbpath)
                
                intrinsic = f['meta']['calibration']['lf_chest_fisheye']['intrinsic'][()]
                distortion = f['meta']['calibration']['lf_chest_fisheye']['distortion'][()]
                camera_meta = {
                    'intrinsic': intrinsic,
                    'distortion': distortion,
                    'camera_model': 'FISHEYE'
                }
                
                points_3d = pointcloud[:, :3].astype(np.float32)
                distances = np.linalg.norm(points_3d, axis=1)
                valid_mask = (
                    np.isfinite(points_3d).all(axis=1) &
                    (distances < MAX_DEPTH) &
                    (points_3d[:, 2] > 0.1)  # Z > 0.1m
                )
                if not valid_mask.any():
                    return depthpath,image
                valid_points = points_3d[valid_mask]
                camera_points = valid_points
                camera_valid_mask = camera_points[:, 2] > 0.01  # Z > 1cm
                if not camera_valid_mask.any():
                    return depthpath, image
                final_valid_points = camera_points[camera_valid_mask]
                x_proj, y_proj, depths, _ = self._project_points_with_distortion(
                    final_valid_points, camera_meta, image.shape[:2]
                )
                if len(x_proj) > 0:
                    # project and visualize
                    projection_img = image.copy()
                    # color by depth
                    normalized_depths = np.clip(depths / MAX_DEPTH, 0, 1)
                    depth_colors = plt.cm.jet(normalized_depths)
                    # draw points
                    for i in range(len(x_proj)):
                        color = (int(depth_colors[i, 2] * 255),  # B
                                int(depth_colors[i, 1] * 255),  # G
                                int(depth_colors[i, 0] * 255))  # R
                        # draw circle
                        cv2.circle(projection_img, (x_proj[i], y_proj[i]), 3, color, -1)
                    cv2.imwrite('projected_depth.png', projection_img)
                    return depthpath, projection_img
                else:
                    return depthpath, image
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def _project_points_with_distortion(self, points: np.ndarray, cam_meta, img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        project 3D points to 2D image plane with distortion
        Args:
            points: points in shape (N, 3)
            cam_meta: intrinsic / distortion parameters / camera model
            img_shape: (h, w) of the image
        Returns:
            Tuple of (x_proj, y_proj, depths, valid_mask)
        """
        # mask points with positive z
        mask_z = points[:, 2] > 0
        z_filtered_points = points[mask_z]
        if len(z_filtered_points) == 0:
            return np.array([]), np.array([]), np.array([]), mask_z
        # compute depths
        depths = np.linalg.norm(z_filtered_points, axis=1)

        x_norm = z_filtered_points[:, 0] / z_filtered_points[:, 2]
        y_norm = z_filtered_points[:, 1] / z_filtered_points[:, 2]

        x_distorted, y_distorted = self._apply_distortion(x_norm, y_norm, cam_meta)
        # project to pixel coordinates
        K = cam_meta['intrinsic']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_proj = (fx * x_distorted + cx).astype(int)
        y_proj = (fy * y_distorted + cy).astype(int)
        # mask out-of-bounds points
        h, w = img_shape
        valid_idx = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
        return x_proj[valid_idx], y_proj[valid_idx], depths[valid_idx], mask_z
    
    def _apply_distortion(self, x_norm: np.ndarray, y_norm: np.ndarray, cam_meta) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x_norm: nomarlized x coordinate
            y_norm: nomarlized y coordinate
            cam_meta: intrinsic / distortion parameters / camera model
        Returns:
            normalized (x_distorted, y_distorted)
        """
        camera_model = cam_meta['camera_model']
        distortion = cam_meta['distortion']
        if camera_model == "FISHEYE" and distortion is not None and len(distortion) >= 4:
            r = np.sqrt(x_norm**2 + y_norm**2)
            theta = np.arctan(r)
            theta_d = theta * (1 + 
                             distortion[0] * theta**2 + 
                             distortion[1] * theta**4 + 
                             distortion[2] * theta**6 + 
                             distortion[3] * theta**8)
            scale = np.ones_like(r)
            mask = r > 1e-8
            scale[mask] = theta_d[mask] / r[mask]
            x_distorted = x_norm * scale
            y_distorted = y_norm * scale
        elif distortion is not None and len(distortion) >= 4:
            r2 = x_norm**2 + y_norm**2
            radial = 1 + distortion[0] * r2 + distortion[1] * r2**2 + distortion[2] * r2**3
            if len(distortion) >= 5:
                dx = 2 * distortion[2] * x_norm * y_norm + distortion[3] * (r2 + 2 * x_norm**2)
                dy = distortion[2] * (r2 + 2 * y_norm**2) + 2 * distortion[3] * x_norm * y_norm
            else:
                dx = 0
                dy = 0
            x_distorted = x_norm * radial + dx
            y_distorted = y_norm * radial + dy
        else:
            x_distorted = x_norm
            y_distorted = y_norm
        return x_distorted, y_distorted
    
    def _convert_poses_to_tracker_format(self, poses: np.ndarray, timestamps: np.ndarray, 
                                       indices: np.ndarray) -> Dict[int, np.ndarray]:
        """
        convert_poses_to_tracker_format
        """
        tracker_data = {}
        
        for i, timestamp in enumerate(timestamps):
            if i < len(poses):
                pose = poses[i]  # [x, y, z, qx, qy, qz, qw]
                
                # create 4x4 transformation matrix
                transform = np.eye(4, dtype=np.float32)
                
                # set translation (x, y, z)
                transform[:3, 3] = pose[:3]
                
                # set rotation (x, y, z, w)
                if len(pose) >= 7:
                    # (x, y, z, w)
                    quat = pose[3:7]
                    # convert to (w, x, y, z) format for scipy
                    quat_scipy = [quat[3], quat[0], quat[1], quat[2]]
                    transform[:3, :3] = R.from_quat([quat_scipy[1], quat_scipy[2], quat_scipy[3], quat_scipy[0]]).as_matrix()
                
                tracker_data[int(timestamp)] = transform
        
        return tracker_data
    
    def _read_tracker_data_from_hdf5(self, hdf5_file) -> Tuple[Optional[Dict], Optional[Dict]]:

        try:
            tracker_data = {}
            tracker_validity = {}
            
            arm_status_available = ('action/arm_status_feedback/timestamp' in hdf5_file)
            
            if arm_status_available:
                arm_timestamps = hdf5_file['action/arm_status_feedback/timestamp'][:]
                arm_indices = hdf5_file['action/arm_status_feedback/index'][:]
                
                if 'action/arm_status_feedback/left_eef_pose_in_chest' in hdf5_file:
                    left_eef_poses = hdf5_file['action/arm_status_feedback/left_eef_pose_in_chest'][:]
                    
                    tracker_data[self.left_tracker_id] = self._convert_poses_to_tracker_format(
                        left_eef_poses, arm_timestamps, arm_indices
                    )
                    
                    if 'action/arm_status_feedback/left_eef_pose_mask' in hdf5_file:
                        left_pose_mask = hdf5_file['action/arm_status_feedback/left_eef_pose_mask'][:]
                        left_validity = {}
                        for i, timestamp in enumerate(arm_timestamps):
                            if i < len(left_pose_mask):
                                left_validity[int(timestamp)] = (left_pose_mask[i] == 0)
                            else:
                                left_validity[int(timestamp)] = True
                        tracker_validity[self.left_tracker_id] = left_validity
                        
                        real_count = sum(1 for v in left_validity.values() if v)
                        total_count = len(left_validity)
                    else:
                        left_validity = {int(ts): True for ts in arm_timestamps}
                        tracker_validity[self.left_tracker_id] = left_validity
                
                if 'action/arm_status_feedback/right_eef_pose_in_chest' in hdf5_file:
                    right_eef_poses = hdf5_file['action/arm_status_feedback/right_eef_pose_in_chest'][:]
                    
                    tracker_data[self.right_tracker_id] = self._convert_poses_to_tracker_format(
                        right_eef_poses, arm_timestamps, arm_indices
                    )
                    
                    if 'action/arm_status_feedback/right_eef_pose_mask' in hdf5_file:
                        right_pose_mask = hdf5_file['action/arm_status_feedback/right_eef_pose_mask'][:]
                        right_validity = {}
                        for i, timestamp in enumerate(arm_timestamps):
                            if i < len(right_pose_mask):
                                right_validity[int(timestamp)] = (right_pose_mask[i] == 0)
                            else:
                                right_validity[int(timestamp)] = True
                        tracker_validity[self.right_tracker_id] = right_validity
                        
                        real_count = sum(1 for v in right_validity.values() if v)
                        total_count = len(right_validity)
                    else:
                        right_validity = {int(ts): True for ts in arm_timestamps}
                        tracker_validity[self.right_tracker_id] = right_validity
            
            if 'action/chest_status_feedback/chest_pose' in hdf5_file:
                chest_poses = hdf5_file['action/chest_status_feedback/chest_pose'][:]
                chest_timestamps = hdf5_file['action/chest_status_feedback/timestamp'][:]
                chest_indices = hdf5_file['action/chest_status_feedback/index'][:]
                
                tracker_data[self.chest_tracker_id] = self._convert_poses_to_tracker_format(
                    chest_poses, chest_timestamps, chest_indices
                )
                
                chest_validity = {int(ts): True for ts in chest_timestamps}
                tracker_validity[self.chest_tracker_id] = chest_validity
                
            
            return (tracker_data if tracker_data else None, 
                   tracker_validity if tracker_validity else None)
            
        except Exception as e:
            return None, None
    
    def _convert_joint_angles_to_glove_format(self, joint_angles: np.ndarray, timestamps: np.ndarray, 
                                            indices: np.ndarray, is_left_hand: bool) -> Dict[int, Dict[str, np.ndarray]]:
        """
        convert_joint_angles_to_glove_format
        Notice that joint_angles actually contains the position data saved from PackToHdf5 (3D position of each bone)
        """
        glove_data = {}
        hand_prefix = "LeftHand_" if is_left_hand else "RightHand_"
        
        # Define the bone names (consistent with finger_bones in PackToHdf5)
        bone_names = [
            # Thumb finger
            f"{hand_prefix}Thumb_Metacarpal",
            f"{hand_prefix}Thumb_Proximal", 
            f"{hand_prefix}Thumb_Distal",
            f"{hand_prefix}Thumb_Tip",
            # Index finger
            f"{hand_prefix}Index_Metacarpal",
            f"{hand_prefix}Index_Proximal",
            f"{hand_prefix}Index_Intermediate",
            f"{hand_prefix}Index_Distal",
            f"{hand_prefix}Index_Tip",
            # Middle finger
            f"{hand_prefix}Middle_Metacarpal",
            f"{hand_prefix}Middle_Proximal",
            f"{hand_prefix}Middle_Intermediate",
            f"{hand_prefix}Middle_Distal",
            f"{hand_prefix}Middle_Tip",
            # Ring finger
            f"{hand_prefix}Ring_Metacarpal",
            f"{hand_prefix}Ring_Proximal",
            f"{hand_prefix}Ring_Intermediate",
            f"{hand_prefix}Ring_Distal",
            f"{hand_prefix}Ring_Tip",
            # Pinky finger
            f"{hand_prefix}Pinky_Metacarpal",
            f"{hand_prefix}Pinky_Proximal",
            f"{hand_prefix}Pinky_Intermediate",
            f"{hand_prefix}Pinky_Distal",
            f"{hand_prefix}Pinky_Tip",
            # Hand
            f"{hand_prefix}Hand"
        ]
        
        for i, timestamp in enumerate(timestamps):

            if joint_angles.shape[1] != 75:
                glove_data[int(timestamp)] = {}
                continue
            
            if i < len(joint_angles):
                position_data = joint_angles[i]
                bones = {}
                
                # each bone has 3 coordinate values (x, y, z)
                for j, bone_name in enumerate(bone_names):
                    start_idx = j * 3
                    end_idx = start_idx + 3
                    
                    if end_idx <= len(position_data):
                        # create 4x4 transformation matrix
                        transform = np.eye(4, dtype=np.float32)
                        transform[:3, 3] = position_data[start_idx:end_idx]
                        bones[bone_name] = transform
                    else:
                        # if data is insufficient, create identity matrix
                        bones[bone_name] = np.eye(4, dtype=np.float32)
                
                glove_data[int(timestamp)] = bones
        
        return glove_data
    
    def _read_glove_data_from_hdf5(self, hdf5_file) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Returns:
            Tuple[glove_data, glove_validity]:
        """
        try:
            glove_data = {}
            glove_validity = {}
            
            if 'action/hand_status_feedback/timestamp' in hdf5_file:
                hand_timestamps = hdf5_file['action/hand_status_feedback/timestamp'][:]
                hand_indices = hdf5_file['action/hand_status_feedback/index'][:]
                
                if 'action/hand_status_feedback/left_hand_joint_angle' in hdf5_file:
                    left_joint_angles = hdf5_file['action/hand_status_feedback/left_hand_joint_angle'][:]
                    
                    glove_data['left_hand'] = self._convert_joint_angles_to_glove_format(
                        left_joint_angles, hand_timestamps, hand_indices, is_left_hand=True
                    )
                    
                    if 'action/hand_status_feedback/left_hand_joint_angle_mask' in hdf5_file:
                        left_hand_mask = hdf5_file['action/hand_status_feedback/left_hand_joint_angle_mask'][:]
                        left_validity = {}
                        for i, timestamp in enumerate(hand_timestamps):
                            if i < len(left_hand_mask):
                                left_validity[int(timestamp)] = (left_hand_mask[i] == 0)
                            else:
                                left_validity[int(timestamp)] = True
                        glove_validity['left_hand'] = left_validity
                        
                        real_count = sum(1 for v in left_validity.values() if v)
                        total_count = len(left_validity)
                    else:
                        left_validity = {int(ts): True for ts in hand_timestamps}
                        glove_validity['left_hand'] = left_validity
                
                if 'action/hand_status_feedback/right_hand_joint_angle' in hdf5_file:
                    right_joint_angles = hdf5_file['action/hand_status_feedback/right_hand_joint_angle'][:]
                    
                    glove_data['right_hand'] = self._convert_joint_angles_to_glove_format(
                        right_joint_angles, hand_timestamps, hand_indices, is_left_hand=False
                    )
                    
                    if 'action/hand_status_feedback/right_hand_joint_angle_mask' in hdf5_file:
                        right_hand_mask = hdf5_file['action/hand_status_feedback/right_hand_joint_angle_mask'][:]
                        right_validity = {}
                        for i, timestamp in enumerate(hand_timestamps):
                            if i < len(right_hand_mask):
                                right_validity[int(timestamp)] = (right_hand_mask[i] == 0)
                            else:
                                right_validity[int(timestamp)] = True
                        glove_validity['right_hand'] = right_validity

                        real_count = sum(1 for v in right_validity.values() if v)
                        total_count = len(right_validity)
                    else:

                        right_validity = {int(ts): True for ts in hand_timestamps}
                        glove_validity['right_hand'] = right_validity
            
            return (glove_data if glove_data else None, 
                   glove_validity if glove_validity else None)
            
        except Exception as e:
            return None, None
        
    def _get_camera_params_from_hdf5(self, timestamp: int, hdf5_file, for_pointcloud_projection: bool = False):

        class CameraParams:
            def __init__(self, hdf5_file, for_pointcloud_projection):
                self.name = "senyun-fpv-lf-fisheye"
                self.camera_model = "fisheye" if not for_pointcloud_projection else "pinhole"
                
                if 'meta/calibration/lf_chest_fisheye/intrinsic' in hdf5_file:
                    self.intrinsic = hdf5_file['meta/calibration/lf_chest_fisheye/intrinsic'][:].astype(np.float32)
                else:
                    self.intrinsic = np.array([
                        [400.0, 0.0, 320.0],
                        [0.0, 400.0, 240.0],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                

                if 'meta/calibration/lf_chest_fisheye/distortion' in hdf5_file:
                    distortion_full = hdf5_file['meta/calibration/lf_chest_fisheye/distortion'][:].astype(np.float32)
                    if for_pointcloud_projection:
                        self.distortion = np.zeros_like(distortion_full)
                    else:
                        self.distortion = distortion_full
                else:
                    self.distortion = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

                if 'meta/calibration/lf_chest_fisheye/extrinsic' in hdf5_file:
                    self.cam2tracker = hdf5_file['meta/calibration/lf_chest_fisheye/extrinsic'][:].astype(np.float32)
                else:

                    self.cam2tracker = np.eye(4, dtype=np.float32)
        
        return CameraParams(hdf5_file, for_pointcloud_projection)
    
    def _record_missing_data_info(self, is_left_hand: bool, 
                                 hand_pose_is_missing: bool,
                                 glove_data_is_missing: bool,
                                 missing_data_info: List[str]) -> None:

        hand_side_en = "Left Hand" if is_left_hand else "Right Hand"
        missing_info_en = []
        
        if hand_pose_is_missing:
            missing_info_en.append("Pose")
        if glove_data_is_missing:
            missing_info_en.append("Glove Data")
        
        missing_data_info.append(f"{hand_side_en} {'/'.join(missing_info_en)} Missing")
        
    def _find_closest_timestamp_in_dict(self, target_timestamp, timestamp_dict) -> Optional[int]:

        if not timestamp_dict:
            return None
        
        timestamps = list(timestamp_dict.keys())
        closest_timestamp = min(timestamps, key=lambda x: abs(x - target_timestamp))
        
        if abs(closest_timestamp - target_timestamp) > 100000:
            return None
            
        return closest_timestamp
        
    def _find_closest_timestamps_for_hand(self, current_timestamp: int, 
                                         hand_type: str, tracker_id: str,
                                         hdf5_glove_data: Dict, 
                                         hdf5_tracker_data: Optional[Dict]) -> Optional[Tuple[int, int, int]]:
        
        glove_timestamp = self._find_closest_timestamp_in_dict(
            current_timestamp, 
            hdf5_glove_data[hand_type]
        )
        tracker_timestamp = self._find_closest_timestamp_in_dict(
            current_timestamp,
            hdf5_tracker_data.get(tracker_id, {}) if hdf5_tracker_data else {}
        )
        chest_timestamp = self._find_closest_timestamp_in_dict(
            current_timestamp,
            hdf5_tracker_data.get(self.chest_tracker_id, {}) if hdf5_tracker_data else {}
        )
        
        if not all([glove_timestamp, tracker_timestamp, chest_timestamp]):
            return None
        
        return glove_timestamp, tracker_timestamp, chest_timestamp
    
    def _check_hand_data_validity(self, hand_type: str, tracker_id: str,
                                 glove_timestamp: int, tracker_timestamp: int,
                                 hdf5_tracker_validity: Optional[Dict],
                                 hdf5_glove_validity: Optional[Dict]) -> Tuple[bool, bool]:
 
        hand_pose_is_missing = False
        glove_data_is_missing = False
        

        if hdf5_tracker_validity and tracker_id in hdf5_tracker_validity:
            hand_validity = hdf5_tracker_validity[tracker_id]
            hand_pose_is_missing = not hand_validity.get(tracker_timestamp, True)
        
        if hdf5_glove_validity and hand_type in hdf5_glove_validity:
            glove_validity = hdf5_glove_validity[hand_type]
            glove_data_is_missing = not glove_validity.get(glove_timestamp, True)
        
        return hand_pose_is_missing, glove_data_is_missing
    
    def _get_wrist2tracker_from_hdf5(self, hdf5_file, is_left_hand: bool) -> Optional[np.ndarray]:

        hand_type = "left_hand" if is_left_hand else "right_hand"
        calibration_path = f'meta/calibration/{hand_type}/extrinsic'
        
        if calibration_path in hdf5_file:
            return hdf5_file[calibration_path][:].astype(np.float32)
        else:
            return None
    
    def _transform_and_project_skeleton(self, glove_bones: Dict, 
                                       hand_tracker_data: np.ndarray,
                                       chest_tracker_data: np.ndarray,
                                       is_left_hand: bool, hdf5_file,
                                       target_camera) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:

        points_2d = {}
        points_validity = {}
        
        wrist2tracker = self._get_wrist2tracker_from_hdf5(hdf5_file, is_left_hand)
        
        for bone_name, wrist_transform in glove_bones.items():
            try:
                if wrist2tracker is not None:
                    tracker_transform = wrist2tracker @ wrist_transform
                else:
                    tracker_transform = wrist_transform
                
                bone_in_chest = hand_tracker_data @ tracker_transform
                

                bone_pose_world = np.linalg.inv(chest_tracker_data) @ bone_in_chest
                

                bone_position_world = bone_pose_world[:3, 3]
                bone_position_world_homogeneous = np.append(bone_position_world, 1.0)
                
                tracker2cam = np.linalg.inv(target_camera.cam2tracker)
                bone_position_cam_homogeneous = tracker2cam @ bone_position_world_homogeneous
                bone_position_cam = bone_position_cam_homogeneous[:3] / bone_position_cam_homogeneous[3]
                
                if bone_position_cam[2] <= 0:
                    continue
                
                point_3d = bone_position_cam.reshape(1, 3)
                
                if not np.isfinite(point_3d).all():
                    continue
                    
                point_2d, point_validity = self._project_to_image(
                    point_3d, target_camera.intrinsic, target_camera.distortion, 
                    target_camera.camera_model
                )
                
                if len(point_2d) == 0 or not point_validity[0]:
                    continue
                    
                points_2d[bone_name] = point_2d[0]
                points_validity[bone_name] = point_validity[0]
                
            except Exception as e:
                continue
        
        return points_2d, points_validity
    
    def _project_to_image(self, points_3d: np.ndarray, intrinsic: np.ndarray, 
                         distortion: np.ndarray, camera_model: str = "fisheye") -> Tuple[np.ndarray, np.ndarray]:

        try:
            if not np.isfinite(points_3d).all():
                points_3d = np.nan_to_num(points_3d, nan=0.0, posinf=0.0, neginf=0.0)
            
            valid_mask = points_3d[:, 2] > 0.01
            if not valid_mask.any():
                empty_result = np.zeros((len(points_3d), 2), dtype=np.float64)
                empty_validity = np.zeros(len(points_3d), dtype=bool)
                return empty_result, empty_validity
            
            rvec = np.zeros(3, dtype=np.float64)
            tvec = np.zeros(3, dtype=np.float64)
            
            result = np.zeros((len(points_3d), 2), dtype=np.float64)
            validity_mask = np.zeros(len(points_3d), dtype=bool)
            
            valid_points = points_3d[valid_mask]
            
            if len(valid_points) > 0:
                if camera_model.lower() == "fisheye":
                    valid_points_2d, _ = cv2.fisheye.projectPoints(
                        valid_points.reshape(-1, 1, 3).astype(np.float64), 
                        rvec, tvec,
                        intrinsic.astype(np.float64), 
                        distortion.astype(np.float64)
                    )
                    valid_points_2d = valid_points_2d.reshape(-1, 2)
                else:
                    valid_points_2d, _ = cv2.projectPoints(
                        valid_points.reshape(-1, 1, 3).astype(np.float64), 
                        rvec, tvec, 
                        intrinsic.astype(np.float64), 
                        distortion.astype(np.float64)
                    )
                    valid_points_2d = valid_points_2d.reshape(-1, 2)

                if np.isfinite(valid_points_2d).all():
                    proj_validity = np.ones(len(valid_points_2d), dtype=bool)
                    
                    final_validity_indices = np.where(valid_mask)[0]
                    validity_mask[final_validity_indices[proj_validity]] = True
                    
                    result[final_validity_indices[proj_validity]] = valid_points_2d[proj_validity]
            
            return result, validity_mask
            
        except Exception as e:
            empty_result = np.zeros((len(points_3d), 2), dtype=np.float64)
            empty_validity = np.zeros(len(points_3d), dtype=bool)
            return empty_result, empty_validity
        
    def _check_skeleton_completeness(self, points_validity: Dict[str, bool], is_left_hand: bool) -> bool:

        hand_prefix = "LeftHand_" if is_left_hand else "RightHand_"
        
        essential_bones = [
            f"{hand_prefix}Hand",
            f"{hand_prefix}Thumb_Distal",
            f"{hand_prefix}Index_Distal", 
            f"{hand_prefix}Middle_Distal",
            f"{hand_prefix}Ring_Distal",
            f"{hand_prefix}Pinky_Distal"
        ]
        
        valid_count = 0
        for bone_name in essential_bones:
            if points_validity.get(bone_name, False):
                valid_count += 1
        return valid_count >= 4 and points_validity.get(f"{hand_prefix}Hand", False)
    
    def _process_and_draw_both_hands(self, image: np.ndarray, frame_idx: int,
                                     current_timestamp: int,
                                     hdf5_glove_data: Dict, hdf5_tracker_data: Dict,
                                     hdf5_file, hdf5_tracker_validity: Optional[Dict],
                                     hdf5_glove_validity: Optional[Dict],
                                     target_camera, missing_data_info: List[str]) -> None:
        for hand_type in ["left_hand", "right_hand"]:
            if hand_type not in hdf5_glove_data:
                continue
            
            is_left_hand = (hand_type == "left_hand")
            
            # Process and draw glove data for a single hand
            self._process_and_draw_single_hand(
                image, frame_idx, hand_type, is_left_hand, current_timestamp,
                hdf5_glove_data, hdf5_tracker_data, hdf5_file,
                hdf5_tracker_validity, hdf5_glove_validity,
                target_camera, missing_data_info
            )
        
    def _process_and_draw_single_hand(self, image: np.ndarray, frame_idx: int,
                                     hand_type: str, is_left_hand: bool, 
                                     current_timestamp: int,
                                     hdf5_glove_data: Dict, hdf5_tracker_data: Dict,
                                     hdf5_file, hdf5_tracker_validity: Optional[Dict],
                                     hdf5_glove_validity: Optional[Dict],
                                     target_camera, missing_data_info: List[str]) -> None:
        
        tracker_id = self.left_tracker_id if is_left_hand else self.right_tracker_id
        
        # 1. Find closest timestamps
        timestamps = self._find_closest_timestamps_for_hand(
            current_timestamp, hand_type, tracker_id,
            hdf5_glove_data, hdf5_tracker_data
        )
        
        if timestamps is None:
            return
        
        glove_timestamp, tracker_timestamp, chest_timestamp = timestamps
        
        # 2. Check data validity
        hand_pose_is_missing, glove_data_is_missing = self._check_hand_data_validity(
            hand_type, tracker_id, glove_timestamp, tracker_timestamp,
            hdf5_tracker_validity, hdf5_glove_validity
        )
        
        # 3. If data is missing, record information and skip drawing
        if hand_pose_is_missing or glove_data_is_missing:
            self._record_missing_data_info(
                is_left_hand, hand_pose_is_missing, 
                glove_data_is_missing, missing_data_info
            )
            return
        
        # 4. Get glove skeleton data
        glove_bones = hdf5_glove_data[hand_type][glove_timestamp]

        hand_tracker_data = hdf5_tracker_data[tracker_id][tracker_timestamp]
        chest_tracker_data = hdf5_tracker_data[self.chest_tracker_id][chest_timestamp]
        
        # 5. Transform and project skeleton to 2D
        points_2d, points_validity = self._transform_and_project_skeleton(
            glove_bones, hand_tracker_data, chest_tracker_data,
            is_left_hand, hdf5_file, target_camera
        )
        
        # 6. Check skeleton completeness and draw
        if self._check_skeleton_completeness(points_validity, is_left_hand):
            self._draw_hand_skeleton(image, points_2d, points_validity, 
                                   is_left_hand, hand_pose_is_missing)
            
    def _draw_hand_skeleton(self, image: np.ndarray, points_2d: Dict[str, np.ndarray], 
                           points_validity: Dict[str, bool], is_left_hand: bool, 
                           pose_is_missing: bool = False) -> np.ndarray:

        hand_connections = [
            ("Hand", "Thumb_Metacarpal"),
            ("Thumb_Metacarpal", "Thumb_Proximal"), 
            ("Thumb_Proximal", "Thumb_Intermediate"),
            ("Thumb_Intermediate", "Thumb_Distal"),
            
            ("Hand", "Index_Metacarpal"),
            ("Index_Metacarpal", "Index_Proximal"),
            ("Index_Proximal", "Index_Intermediate"),
            ("Index_Intermediate", "Index_Distal"),
            

            ("Hand", "Middle_Metacarpal"),
            ("Middle_Metacarpal", "Middle_Proximal"),
            ("Middle_Proximal", "Middle_Intermediate"),
            ("Middle_Intermediate", "Middle_Distal"),
            
            ("Hand", "Ring_Metacarpal"),
            ("Ring_Metacarpal", "Ring_Proximal"),
            ("Ring_Proximal", "Ring_Intermediate"),
            ("Ring_Intermediate", "Ring_Distal"),
            
            ("Hand", "Pinky_Metacarpal"),
            ("Pinky_Metacarpal", "Pinky_Proximal"),
            ("Pinky_Proximal", "Pinky_Intermediate"),
            ("Pinky_Intermediate", "Pinky_Distal"),
        ]
        
        if pose_is_missing:
            color = (0, 0, 255)
            hand_line_thickness = 8
            hand_point_radius = 14
        else:
            color = (255, 0, 0)
            hand_line_thickness = 10
            hand_point_radius = 16
        
        for start_bone, end_bone in hand_connections:
            hand_prefix = "LeftHand_" if is_left_hand else "RightHand_"
            start_name = hand_prefix + start_bone
            end_name = hand_prefix + end_bone
            
            if (start_name in points_2d and end_name in points_2d and 
                points_validity.get(start_name, False) and points_validity.get(end_name, False)):
                
                start_point = points_2d[start_name]
                end_point = points_2d[end_name]
                
                if (np.isfinite(start_point).all() and len(start_point) == 2 and
                    np.isfinite(end_point).all() and len(end_point) == 2):
                    
                    start_x, start_y = int(start_point[0]), int(start_point[1])
                    end_x, end_y = int(end_point[0]), int(end_point[1])
                    
                    if (0 <= start_x < image.shape[1] and 0 <= start_y < image.shape[0] and
                        0 <= end_x < image.shape[1] and 0 <= end_y < image.shape[0]):
                        
                        cv2.line(image, (start_x, start_y), (end_x, end_y), color, hand_line_thickness)
        
        connected_bones = set()
        for start_bone, end_bone in hand_connections:
            hand_prefix = "LeftHand_" if is_left_hand else "RightHand_"
            connected_bones.add(hand_prefix + start_bone)
            connected_bones.add(hand_prefix + end_bone)
        
        for bone_name, point_2d in points_2d.items():
            if (is_left_hand and "LeftHand" in bone_name) or \
               (not is_left_hand and "RightHand" in bone_name):
                
                if points_validity.get(bone_name, False) and bone_name in connected_bones:

                    if np.isfinite(point_2d).all() and len(point_2d) == 2:
                        x, y = int(point_2d[0]), int(point_2d[1])

                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            cv2.circle(image, (x, y), hand_point_radius, color, -1)
        
        if pose_is_missing:
            hand_side = "Left Hand" if is_left_hand else "Right Hand"
            text = f"{hand_side} Pose Missing"
            
            if is_left_hand:
                text_position = (50, 80)
            else:
                text_position = (image.shape[1] - 200, 80)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            overlay = image.copy()
            cv2.rectangle(overlay, 
                         (text_position[0] - 10, text_position[1] - text_size[1] - 10),
                         (text_position[0] + text_size[0] + 10, text_position[1] + 10),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            cv2.putText(image, text, text_position, font, font_scale, (0, 0, 255), thickness)
                
        return image
    
    def _draw_missing_data_notification(self, image: np.ndarray, missing_info: List[str], frame_idx: int) -> np.ndarray:
        """
        Draw aesthetically pleasing data missing notification on image
        
        Args:
            image: Input image
            missing_info: Missing information list, e.g. ["Left hand pose missing", "Right hand glove data missing"]
            frame_idx: Current frame index
            
        Returns:
            Image with notification drawn
        """
        if not missing_info:
            return image
        
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Set notification area position and size (enlarged text box)
        notification_width = min(w - 40, 800)  # Maximum width 800 pixels, 20px margin on each side
        notification_height = 80 + len(missing_info) * 50  # Base height 80 + 50px per info item
        
        # Notification box position (centered top)
        box_x = (w - notification_width) // 2
        box_y = 30
        
        # Draw semi-transparent background box
        cv2.rectangle(overlay, 
                     (box_x, box_y), 
                     (box_x + notification_width, box_y + notification_height),
                     (0, 0, 0), -1)  # Black background
        
        # Draw border (wider border)
        cv2.rectangle(overlay, 
                     (box_x, box_y), 
                     (box_x + notification_width, box_y + notification_height),
                     (0, 0, 255), 5)  # Red border, increased from 3 to 5
        
        # Apply transparency
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Set font parameters (larger text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_font_scale = 1.6  # Increased from 1.2 to 1.6
        info_font_scale = 1.2   # Increased from 0.9 to 1.2
        title_thickness = 3     # Increased from 2 to 3
        info_thickness = 3      # Increased from 2 to 3
        
        # Draw title
        title_text = f"Frame {frame_idx} Data Missing Warning"
        title_size = cv2.getTextSize(title_text, font, title_font_scale, title_thickness)[0]
        title_x = box_x + (notification_width - title_size[0]) // 2
        title_y = box_y + 45  # Increased from 35 to 45 to accommodate larger text box
        
        cv2.putText(image, title_text, (title_x, title_y), 
                   font, title_font_scale, (0, 255, 255), title_thickness)  # Yellow title
        
        # Draw separator line
        line_y = title_y + 10
        cv2.line(image, (box_x + 20, line_y), (box_x + notification_width - 20, line_y), 
                (255, 255, 255), 1)
        
        # Draw missing information list
        for i, info in enumerate(missing_info):
            info_y = line_y + 40 + i * 50  # Adjusted from 30+i*35 to 40+i*50 for larger line spacing
            info_x = box_x + 40  # Increased from 30 to 40, increased left margin
            
            # Draw bullet point (enlarged bullet)
            cv2.circle(image, (info_x - 15, info_y - 10), 6, (255, 100, 100), -1)  # Increased from 4px to 6px
            
            # Draw information text
            cv2.putText(image, info, (info_x, info_y), 
                       font, info_font_scale, (255, 255, 255), info_thickness)  # White text
        
        return image
    
    def _draw_skeleton_mask(self, mask: np.ndarray, all_skeleton_points: Dict[str, Dict[str, np.ndarray]], 
                           all_skeleton_validity: Dict[str, Dict[str, bool]]) -> None:
        """
        Draw hand skeleton (white) on mask
        
        Args:
            mask: Mask image (will be modified directly)
            all_skeleton_points: All hands skeleton projection points dict {hand_type: {bone_name: point_2d}}
            all_skeleton_validity: All hands skeleton validity dict {hand_type: {bone_name: validity}}
        """
        # Define hand bone connections
        hand_connections = [
            # Thumb (Thumb only has Metacarpal, Proximal, Distal, Tip)
            ("Hand", "Thumb_Metacarpal"),
            ("Thumb_Metacarpal", "Thumb_Proximal"), 
            ("Thumb_Proximal", "Thumb_Distal"),
            ("Thumb_Distal", "Thumb_Tip"),
            
            # Index finger
            ("Hand", "Index_Metacarpal"),
            ("Index_Metacarpal", "Index_Proximal"),
            ("Index_Proximal", "Index_Intermediate"),
            ("Index_Intermediate", "Index_Distal"),
            ("Index_Distal", "Index_Tip"),
            
            # Middle finger
            ("Hand", "Middle_Metacarpal"),
            ("Middle_Metacarpal", "Middle_Proximal"),
            ("Middle_Proximal", "Middle_Intermediate"),
            ("Middle_Intermediate", "Middle_Distal"),
            ("Middle_Distal", "Middle_Tip"),
            
            # Ring finger
            ("Hand", "Ring_Metacarpal"),
            ("Ring_Metacarpal", "Ring_Proximal"),
            ("Ring_Proximal", "Ring_Intermediate"),
            ("Ring_Intermediate", "Ring_Distal"),
            ("Ring_Distal", "Ring_Tip"),
            
            # Pinky finger
            ("Hand", "Pinky_Metacarpal"),
            ("Pinky_Metacarpal", "Pinky_Proximal"),
            ("Pinky_Proximal", "Pinky_Intermediate"),
            ("Pinky_Intermediate", "Pinky_Distal"),
            ("Pinky_Distal", "Pinky_Tip"),
        ]
        
        # White color for drawing on mask
        white = 255
        
        # Line thickness and point radius
        line_thickness = 20  # Thicker lines to cover larger area after dilation
        point_radius = 25    # Larger point radius
        
        # Draw skeleton for each hand
        for hand_type, points_2d in all_skeleton_points.items():
            points_validity = all_skeleton_validity.get(hand_type, {})
            is_left_hand = (hand_type == "left_hand")
            hand_prefix = "LeftHand_" if is_left_hand else "RightHand_"
            
            # Draw connecting lines
            for start_bone, end_bone in hand_connections:
                start_name = hand_prefix + start_bone
                end_name = hand_prefix + end_bone
                
                # Check if both points exist and are valid
                if (start_name in points_2d and end_name in points_2d and 
                    points_validity.get(start_name, False) and points_validity.get(end_name, False)):
                    
                    start_point = points_2d[start_name]
                    end_point = points_2d[end_name]
                    
                    # Check if coordinates of both points are valid
                    if (np.isfinite(start_point).all() and len(start_point) == 2 and
                        np.isfinite(end_point).all() and len(end_point) == 2):
                        
                        start_x, start_y = int(start_point[0]), int(start_point[1])
                        end_x, end_y = int(end_point[0]), int(end_point[1])
                        
                        # Check if coordinates are within image bounds
                        if (0 <= start_x < mask.shape[1] and 0 <= start_y < mask.shape[0] and
                            0 <= end_x < mask.shape[1] and 0 <= end_y < mask.shape[0]):
                            
                            cv2.line(mask, (start_x, start_y), (end_x, end_y), white, line_thickness)
            
            # Collect all bone point names that appear in connection relationships
            connected_bones = set()
            for start_bone, end_bone in hand_connections:
                connected_bones.add(hand_prefix + start_bone)
                connected_bones.add(hand_prefix + end_bone)
            
            # Draw bone points
            for bone_name, point_2d in points_2d.items():
                # Only draw valid points that appear in connection relationships
                if points_validity.get(bone_name, False) and bone_name in connected_bones:
                    # Check if coordinates are valid
                    if np.isfinite(point_2d).all() and len(point_2d) == 2:
                        x, y = int(point_2d[0]), int(point_2d[1])
                        # Check if coordinates are within image bounds
                        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                            cv2.circle(mask, (x, y), point_radius, white, -1)
                            
    def _read_data_quality_from_hdf5(self, hdf5_file, frame_idx: int, indices_data: Dict = None) -> Optional[str]:
        """
        Read data_quality data from HDF5 file
        
        Args:
            hdf5_file: HDF5 file object
            frame_idx: Frame index (corresponding to main camera index)
            indices_data: Index data dictionary for finding arm_status_feedback corresponding index
            
        Returns:
            Optional[str]: data_quality status string, returns None if not exists
        """
        try:
            if 'action/arm_status_feedback/data_quality' not in hdf5_file:
                return None
            
            data_quality_array = hdf5_file['action/arm_status_feedback/data_quality']
            
            # If indices_data is provided, try to find arm_status_feedback index via main camera index
            if indices_data and 'lf_chest_fisheye' in indices_data:
                camera_data = indices_data['lf_chest_fisheye']
                if frame_idx < len(camera_data['indices']):
                    main_camera_index = camera_data['indices'][frame_idx]
                    # Read arm_status_feedback index mapping
                    if 'action/arm_status_feedback/index' in hdf5_file:
                        arm_indices = hdf5_file['action/arm_status_feedback/index'][:]
                        # Find arm index closest to main_camera_index
                        if len(arm_indices) > 0:
                            # Find index position closest to main_camera_index
                            arm_idx = np.argmin(np.abs(arm_indices - main_camera_index))
                            if arm_idx < len(data_quality_array):
                                quality = data_quality_array[arm_idx]
                                return quality.decode('utf-8') if isinstance(quality, bytes) else str(quality)
            
            # If cannot align via index, directly use frame_idx (assume data_quality has same frame count as main camera)
            if frame_idx < len(data_quality_array):
                quality = data_quality_array[frame_idx]
                return quality.decode('utf-8') if isinstance(quality, bytes) else str(quality)
            
            return None
        except Exception as e:
            print(f"Error reading data_quality: {e}")
            return None
        
    def _draw_data_quality_notification(self, image: np.ndarray, data_quality: str) -> np.ndarray:
        """
        Draw data_quality information on image
        
        Args:
            image: Input image
            data_quality: Data quality status string
            
        Returns:
            Image with data_quality information drawn
        """
        if not data_quality:
            return image
        
        h, w = image.shape[:2]
        
        # Set font and size (larger font for better visibility)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 6
        
        # Select color based on status
        if "INCOMPLETE" in data_quality.upper():
            color = (0, 0, 255)  # Red
        elif "COMPLETE" in data_quality.upper() or "GOOD" in data_quality.upper():
            color = (0, 255, 0)  # Green
        else:
            color = (0, 255, 255)  # Yellow
        
        # Prepare text
        text = f"Data Quality: {data_quality}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Set position (top left corner with some margin)
        x = 20
        y = text_height + 30
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, 
                     (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + baseline + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return image
              
    def generate_glove_visualization(self, scene, eps, index) -> Optional[np.ndarray]:

        h5_path = os.path.join(self.dataroot, scene, eps, "dataset.hdf5")
        glove_data = {}
        glove_validity = {}
        with h5py.File(h5_path, 'r') as f:
            indices_data = f['observation']['camera']
            rgbpath = f['observation']['camera']['lf_chest_fisheye']['filepath'][()][index]
            rgbpath = os.path.join(self.dataroot, scene, eps, rgbpath.decode('utf-8'))
            
            image = cv2.imread(rgbpath)
            if image is None:
                return None
            
            glove_data, glove_validity = self._read_glove_data_from_hdf5(f)
            hdf5_tracker_data, hdf5_tracker_validity = self._read_tracker_data_from_hdf5(f)
            if not glove_data:
                return image

            current_timestamp = indices_data['lf_chest_fisheye']['timestamp'][index]
            target_camera = self._get_camera_params_from_hdf5(current_timestamp, f)
            missing_data_info = []
            skeleton_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            all_skeleton_points = {}
            all_skeleton_validity = {}

            for hand_type in ["left_hand", "right_hand"]:
                if hand_type not in glove_data:
                    continue

                is_left_hand = (hand_type == "left_hand")
                tracker_id = self.left_tracker_id if is_left_hand else self.right_tracker_id

                timestamps = self._find_closest_timestamps_for_hand(
                    current_timestamp, hand_type, tracker_id,
                    glove_data, hdf5_tracker_data
                )
                if timestamps is None:
                    continue
                glove_timestamp, tracker_timestamp, chest_timestamp = timestamps
                hand_pose_is_missing, glove_data_is_missing = self._check_hand_data_validity(
                    hand_type, tracker_id, glove_timestamp, tracker_timestamp,
                    hdf5_tracker_validity, glove_validity
                )

                if hand_pose_is_missing or glove_data_is_missing:
                    self._record_missing_data_info(
                        is_left_hand, hand_pose_is_missing, 
                        glove_data_is_missing, missing_data_info
                    )
                    continue
                glove_bones = glove_data[hand_type][glove_timestamp]
                hand_tracker_data = hdf5_tracker_data[tracker_id][tracker_timestamp]
                chest_tracker_data = hdf5_tracker_data[self.chest_tracker_id][chest_timestamp]

                points_2d, points_validity = self._transform_and_project_skeleton(
                    glove_bones, hand_tracker_data, chest_tracker_data,
                    is_left_hand, f, target_camera
                )
                # save projected points for skeleton mask drawing
                if self._check_skeleton_completeness(points_validity, is_left_hand):
                    all_skeleton_points[hand_type] = points_2d
                    all_skeleton_validity[hand_type] = points_validity

            # 7. draw skeleton on mask
            self._draw_skeleton_mask(skeleton_mask, all_skeleton_points, all_skeleton_validity)

            # 8. mask dilation 
            kernel_size = 15 # kernal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_mask = cv2.dilate(skeleton_mask, kernel, iterations=1)

            # 9. add mask to image
            mask_alpha = 0.3  # transparency
            white_color = np.array([255, 255, 255], dtype=np.uint8)
            mask_overlay = np.zeros_like(image)
            mask_overlay[dilated_mask > 0] = white_color
            image = cv2.addWeighted(image, 1.0, mask_overlay, mask_alpha, 0)

            # 10. process and draw both hands
            self._process_and_draw_both_hands(
                image, index, current_timestamp,
                glove_data, hdf5_tracker_data, f,
                hdf5_tracker_validity, glove_validity,
                target_camera, missing_data_info
            )

            # 11. draw enhanced current wrist point
            image = self._draw_wrist_trajectory(
                image, index, indices_data, 
                glove_data, hdf5_tracker_data, 
                target_camera, f, hdf5_tracker_validity
            )

            # 13. show missing data notification
            if missing_data_info:
                image = self._draw_missing_data_notification(
                    image, missing_data_info, index
                )

            # 14. check data quality and draw notification
            if f is not None:
                data_quality = self._read_data_quality_from_hdf5(
                    f, index, indices_data
                )
                if data_quality:
                    image = self._draw_data_quality_notification(image, data_quality)
            return image
        
    def _draw_alpha_line(self, image: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], 
                        color: Tuple[int, int, int], thickness: int, alpha: float) -> None:
        """draw semi-transparent line"""
        overlay = image.copy()
        cv2.line(overlay, start, end, color, thickness)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    def _draw_alpha_circle(self, image: np.ndarray, center: Tuple[int, int], radius: int, 
                          color: Tuple[int, int, int], alpha: float, thickness: int = -1) -> None:
        """draw semi-transparent circle"""
        overlay = image.copy()
        cv2.circle(overlay, center, radius, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
    def _is_point_in_image(self, point: np.ndarray, image: np.ndarray) -> bool:
        """check if point is within image"""
        return (0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0])
        
    def _draw_enhanced_current_wrist_point(self, image: np.ndarray, current_point: Dict) -> None:
        """
        Draw enhanced current wrist point visualization
        
        Args:
            image: Image to draw on
            current_point: Current frame wrist point data
        """
        pos = current_point['position'].astype(int)
        if not self._is_point_in_image(pos, image):
            return
        
        center = tuple(pos)
        alpha = current_point.get('alpha', 1.0)
        
        # Draw multi-layered wrist point visualization 
        # 1. Outer layer: White halo (large)
        self._draw_alpha_circle(image, center, 25, (255, 255, 255), alpha * 0.15)
        
        # 2. Middle outer layer: Light red halo (medium)
        self._draw_alpha_circle(image, center, 20, (128, 128, 255), alpha * 0.3)
        
        # 3. Middle layer: Red circle (border)
        self._draw_alpha_circle(image, center, 15, (0, 0, 255), alpha * 0.8, thickness=3)
        
        # 4. Inner layer: Solid red circle
        self._draw_alpha_circle(image, center, 12, (0, 0, 255), alpha * 0.9)
        
        # 5. Center: White crosshair
        cross_size = 8
        cross_thickness = 2
        # Horizontal line
        start_h = (center[0] - cross_size, center[1])
        end_h = (center[0] + cross_size, center[1])
        self._draw_alpha_line(image, start_h, end_h, (255, 255, 255), cross_thickness, alpha)
        
        # Vertical line
        start_v = (center[0], center[1] - cross_size)
        end_v = (center[0], center[1] + cross_size)
        self._draw_alpha_line(image, start_v, end_v, (255, 255, 255), cross_thickness, alpha)
        
    def _generate_linear_interpolation_aligned(self, wrist_points: List[Dict], current_frame_idx: int = None) -> List[Dict]:        
        """
        Generate aligned linear interpolation points (backup solution when there are too few points)
        Ensure current frame points are correctly aligned
        
        Args:
            wrist_points: Original trajectory point list
            current_frame_idx: Index of current frame in the list
            
        Returns:
            Trajectory point list containing linear interpolation points
        """
        if len(wrist_points) < 2:
            return wrist_points
            
        dense_points = []
        
        # Add all original trajectory points
        for point_data in wrist_points:
            dense_points.append(point_data)
        
        # Generate linear interpolation points between adjacent frames
        for i in range(len(wrist_points) - 1):
            start_point = wrist_points[i]
            end_point = wrist_points[i + 1]
            
            start_pos = start_point['position']
            end_pos = end_point['position']
            
            # Calculate number of interpolation points (increase density if current frame is involved)
            distance = np.linalg.norm(end_pos - start_pos)
            
            is_current_segment = (current_frame_idx is not None and 
                                (i == current_frame_idx or i + 1 == current_frame_idx))
            
            if is_current_segment:
                # Use higher density near current frame
                num_interpolations = max(2, min(10, int(distance / 3)))
            else:
                # Use normal density for other line segments
                num_interpolations = max(1, min(8, int(distance / 5)))
            
            # Linear interpolation
            for j in range(1, num_interpolations):
                t = j / num_interpolations
                interpolated_pos = start_pos * (1 - t) + end_pos * t
                interpolated_alpha = start_point['alpha'] * (1 - t) + end_point['alpha'] * t
                
                # Interpolate frame_idx
                interpolated_frame_idx = start_point['frame_idx'] * (1 - t) + end_point['frame_idx'] * t
                
                # Check if close to current frame
                is_near_current = False
                if current_frame_idx is not None:
                    current_point = wrist_points[current_frame_idx]
                    current_real_frame = current_point['frame_idx']
                    
                    # Mark as current frame related if interpolated point's frame_idx is close to current frame's frame_idx
                    frame_distance = abs(interpolated_frame_idx - current_real_frame)
                    is_near_current = frame_distance < 0.5
                
                # Determine if it's a future frame interpolation point
                is_future_interpolated = interpolated_frame_idx > current_real_frame if current_frame_idx is not None else False
                
                dense_points.append({
                    'position': interpolated_pos,
                    'alpha': interpolated_alpha,
                    'frame_idx': interpolated_frame_idx,
                    'is_interpolated': True,
                    'is_current': is_near_current,
                    'is_future': is_future_interpolated
                })
        
        # Sort by frame_idx
        dense_points.sort(key=lambda x: x['frame_idx'])
        
        return dense_points
    
    def _catmull_rom_interpolate(self, p0: np.ndarray, p1: np.ndarray, 
                                p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
        """
        Catmull-Rom spline interpolation
        
        Args:
            p0, p1, p2, p3: Control points
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated point position
        """
        t2 = t * t
        t3 = t2 * t
        
        # Catmull-Rom formula
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        
    def _generate_smooth_trajectory_points(self, wrist_points: List[Dict]) -> List[Dict]:
        """
        Generate dense trajectory points using Catmull-Rom spline interpolation (reference UpdateGloveData implementation)
        Ensure current frame points are correctly aligned in trajectory
        
        Args:
            wrist_points: Original trajectory point list
            
        Returns:
            Dense trajectory point list containing interpolation points
        """
        if len(wrist_points) < 2:
            # Too few points, return original points directly
            return wrist_points
        
        # Find index of current frame point
        current_frame_idx = None
        for i, point in enumerate(wrist_points):
            if point.get('is_current', False):
                current_frame_idx = i
                break
        
        dense_points = []
        
        # Add all original trajectory points (ensure current frame points are correctly marked)
        for i, point_data in enumerate(wrist_points):
            # Reconfirm current frame marking
            point_copy = point_data.copy()
            point_copy['is_current'] = (i == current_frame_idx)
            # Ensure is_future mark exists
            if 'is_future' not in point_copy:
                point_copy['is_future'] = point_copy.get('frame_idx', 0) > (wrist_points[current_frame_idx]['frame_idx'] if current_frame_idx is not None else 0)
            dense_points.append(point_copy)
        
        # If less than 3 points, use linear interpolation
        if len(wrist_points) < 3:
            return self._generate_linear_interpolation_aligned(dense_points, current_frame_idx)
        
        # Use Catmull-Rom spline for smooth interpolation
        for i in range(len(wrist_points) - 1):
            start_idx = max(0, i - 1)
            end_idx = min(len(wrist_points) - 1, i + 2)
            
            # Get control points
            p0 = wrist_points[start_idx]['position']
            p1 = wrist_points[i]['position']
            p2 = wrist_points[i + 1]['position']
            p3 = wrist_points[end_idx]['position']
            
            # Calculate number of interpolation points (adjust based on distance and proximity to current frame)
            distance = np.linalg.norm(p2 - p1)
            
            # Increase interpolation density if this segment contains current frame
            is_current_segment = (current_frame_idx is not None and 
                                (i == current_frame_idx or i + 1 == current_frame_idx))
            
            if is_current_segment:
                # Use higher density for line segments near current frame
                num_interpolations = max(3, min(15, int(distance / 2)))
            else:
                # Use normal density for other line segments
                num_interpolations = max(2, min(12, int(distance / 3)))
            
            # Generate Catmull-Rom interpolation points
            for j in range(1, num_interpolations):
                t = j / num_interpolations
                
                # Catmull-Rom spline interpolation
                interpolated_pos = self._catmull_rom_interpolate(p0, p1, p2, p3, t)
                
                # Interpolate other attributes
                start_point = wrist_points[i]
                end_point = wrist_points[i + 1]
                
                interpolated_alpha = start_point['alpha'] * (1 - t) + end_point['alpha'] * t
                
                # Interpolate frame_idx
                interpolated_frame_idx = start_point['index'] * (1 - t) + end_point['index'] * t
                
                # Check if close to current frame (more precise judgment)
                is_near_current = False
                if current_frame_idx is not None:
                    current_point = wrist_points[current_frame_idx]
                    current_real_frame = current_point['index']
                    
                    # Mark as current frame related if interpolated point's frame_idx is close to current frame's frame_idx
                    frame_distance = abs(interpolated_frame_idx - current_real_frame)
                    is_near_current = frame_distance < 0.5
                
                # Determine if it's a future frame interpolation point
                is_future_interpolated = interpolated_frame_idx > current_point['index'] if current_frame_idx is not None else False
                
                dense_points.append({
                    'position': interpolated_pos,
                    'alpha': interpolated_alpha,
                    'index': interpolated_frame_idx,
                    'is_interpolated': True,
                    'is_current': is_near_current,
                    'is_future': is_future_interpolated
                })
        
        # Sort by frame_idx to ensure correct trajectory order
        dense_points.sort(key=lambda x: x['index'])
        
        return dense_points
    
    def _draw_trajectory_lines(self, image: np.ndarray, wrist_points: List[Dict], current_point: Dict = None) -> None:
        """
        Draw lines connecting trajectory points, ensuring the trajectory is centered on the current frame
        
        Args:
            image: Image (will be modified directly)
            wrist_points: List of wrist trajectory points
            current_point: Current frame wrist point (optional)
        """
        if len(wrist_points) < 2:
            return
        
        # Find the position of current frame in the trajectory
        current_frame_idx = None
        if current_point:
            for i, point in enumerate(wrist_points):
                if point.get('is_current', False):
                    current_frame_idx = i
                    break
        
        # Draw trajectory line segments
        for i in range(len(wrist_points) - 1):
            start_point = wrist_points[i]
            end_point = wrist_points[i + 1]
            
            start_pos = start_point['position'].astype(int)
            end_pos = end_point['position'].astype(int)
            
            # Check if both points are within image boundaries
            if (self._is_point_in_image(start_pos, image) and 
                self._is_point_in_image(end_pos, image)):
                
                # Calculate line segment transparency (based on average transparency of both endpoints)
                avg_alpha = (start_point.get('alpha', 1.0) + end_point.get('alpha', 1.0)) / 2
                
                # Select color and thickness based on whether it contains the current frame point
                if current_frame_idx is not None and (i == current_frame_idx or i + 1 == current_frame_idx):
                    # Lines connecting to current frame use red, thicker
                    line_color = (0, 0, 255)  # Red
                    thickness = 6
                    alpha = avg_alpha * 0.8
                else:
                    # Future trajectory lines use green, thinner
                    line_color = (0, 200, 0)  # Green
                    thickness = 4
                    alpha = avg_alpha * 0.6
                
                # Draw line segment
                self._draw_alpha_line(image, tuple(start_pos), tuple(end_pos), 
                                    line_color, thickness, alpha)
        
    def _render_simple_wrist_trajectory(self, image: np.ndarray, wrist_points: List[Dict]) -> None:
        """
        Render wrist trajectory onto image (using dense sampled points and time-based color gradient, reference UpdateGloveData)
        
        Args:
            image: Image (will be modified directly)
            wrist_points: List of wrist trajectory points
        """
        if len(wrist_points) == 0:
            return
            
        # Sort by frame index
        wrist_points.sort(key=lambda x: x['index'])
        
        # Find wrist point of current frame
        current_wrist_point = None
        for point in wrist_points:
            if point.get('is_current', False):
                current_wrist_point = point
                break
        
        # If there's only one point (current frame), still display it
        if len(wrist_points) == 1:
            if current_wrist_point:
                self._draw_enhanced_current_wrist_point(image, current_wrist_point)
            return
        
        # Current frame uses bright red, more prominent
        current_color = (0, 0, 255)           # Bright red (BGR)
        current_outline_color = (255, 255, 255)  # White outline (BGR)
        
        # Future frames use time-based gradient colors: bright green -> dark green
        # Later time (larger frame index) has darker color
        early_future_color = (0, 255, 0)      # Bright green (BGR) - near future
        late_future_color = (0, 128, 0)       # Dark green (BGR) - far future
        
        # Generate smooth dense sampling points
        dense_points = self._generate_smooth_trajectory_points(wrist_points)
        
        # Calculate time range (for time-based color gradient)
        if len(wrist_points) > 1:
            min_frame_idx = min(point['index'] for point in wrist_points)
            max_frame_idx = max(point['index'] for point in wrist_points)
            frame_range = max_frame_idx - min_frame_idx
        else:
            min_frame_idx = max_frame_idx = frame_range = 0
        
        # First draw trajectory lines (connecting adjacent original trajectory points)
        self._draw_trajectory_lines(image, wrist_points, current_wrist_point)
        
        # Draw all dense sampling points
        for point_data in dense_points:
            pos = point_data['position'].astype(int)
            if self._is_point_in_image(pos, image):
                alpha = point_data['alpha']
                is_current = point_data.get('is_current', False)
                is_interpolated = point_data.get('is_interpolated', False)
                frame_idx = point_data.get('index', 0)
                
                # Select color and size
                if is_current and not is_interpolated:
                    # Current frame original point - skip, will be drawn separately later
                    continue
                elif is_current:
                    # Current frame interpolated point uses red, slightly smaller
                    color = current_color
                    radius = 6
                else:
                    # Future frames use time-based gradient colors (green series)
                    color = self._get_time_based_color(
                        frame_idx, min_frame_idx, frame_range, early_future_color, late_future_color
                    )
                    radius = 5
                
                # Draw dense sampling point
                self._draw_alpha_circle(image, tuple(pos), radius, color, alpha)
        
        # Finally draw current frame wrist point, ensuring it's on top layer and most prominent
        if current_wrist_point:
            self._draw_enhanced_current_wrist_point(image, current_wrist_point)
            
    def _get_time_based_color(self, frame_idx: int, min_frame_idx: int, frame_range: int,
                             early_color: Tuple[int, int, int], 
                             late_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Generate gradient color based on time sequence (reference UpdateGloveData implementation)
        
        Args:
            frame_idx: Current frame index
            min_frame_idx: Minimum frame index in chunk
            frame_range: Frame range of chunk
            early_color: Color at earlier time (light color)
            late_color: Color at later time (dark color)
            
        Returns:
            Time-based gradient color (BGR)
        """
        if frame_range == 0:
            return early_color
        
        # Normalize time position to [0, 1] range, later time has larger value
        normalized_time = (frame_idx - min_frame_idx) / frame_range
        normalized_time = max(0.0, min(1.0, normalized_time))
        
        # Linear interpolation to calculate color
        color = tuple(
            int(early_color[i] * (1 - normalized_time) + late_color[i] * normalized_time)
            for i in range(3)
        )
        
        return color
            
    def _draw_wrist_trajectory(self, image: np.ndarray, current_index: int, 
                              indices_data: Dict, glove_data: Optional[Dict],
                              hdf5_tracker_data: Optional[Dict], target_camera,
                              hdf5_file, hdf5_tracker_validity: Optional[Dict] = None) -> np.ndarray:
        """
        Draw wrist trajectory on the current frame

        Args:
            image: Current frame image
            current_index: Current frame index
            indices_data: Index data
            glove_data: HDF5 glove data
            hdf5_tracker_data: HDF5 tracker data
            target_camera: Target camera object
            hdf5_file: HDF5 file object
            hdf5_tracker_validity: Tracker validity mask

        Returns:
            Image with wrist trajectory drawn
        """
        if not glove_data or not hdf5_tracker_data:
            return image

        # Get main camera data
        if 'lf_chest_fisheye' not in indices_data:
            return image

        camera_data = indices_data['lf_chest_fisheye']
        total_frames = len(camera_data['timestamp'])

        # Calculate trajectory range: only show trajectory from current frame to future
        trajectory_start = current_index  # Start from current frame
        trajectory_end = min(total_frames, current_index + self.trajectory_range + 1)  # To future frames

        # Process wrist trajectory for left and right hands separately
        for hand_type in ["left_hand", "right_hand"]:
            if hand_type not in glove_data:
                continue

            is_left_hand = (hand_type == "left_hand")
            tracker_id = self.left_tracker_id if is_left_hand else self.right_tracker_id

            # Collect wrist trajectory points
            wrist_points = []

            for index in range(trajectory_start, trajectory_end):
                if index >= len(camera_data['timestamp']):
                    continue
                # Get timestamp of current frame
                frame_timestamp = camera_data['timestamp'][index]
                # Find closest timestamps
                glove_timestamp = self._find_closest_timestamp_in_dict(
                    frame_timestamp, 
                    glove_data[hand_type]
                )
                tracker_timestamp = self._find_closest_timestamp_in_dict(
                    frame_timestamp,
                    hdf5_tracker_data.get(tracker_id, {})
                )
                chest_timestamp = self._find_closest_timestamp_in_dict(
                    frame_timestamp,
                    hdf5_tracker_data.get(self.chest_tracker_id, {})
                )
                if not all([glove_timestamp, tracker_timestamp, chest_timestamp]):
                    continue
                # Check tracker data validity (eef_pose_mask=0 means valid, =1 means invalid)
                if hdf5_tracker_validity:
                    # Check if hand tracker data is valid
                    if tracker_id in hdf5_tracker_validity:
                        hand_validity = hdf5_tracker_validity[tracker_id]
                        if not hand_validity.get(tracker_timestamp, True):
                            # Hand pose invalid, skip this frame
                            continue
                    # Check if chest tracker data is valid
                    if self.chest_tracker_id in hdf5_tracker_validity:
                        chest_validity = hdf5_tracker_validity[self.chest_tracker_id]
                        if not chest_validity.get(chest_timestamp, True):
                            # Chest pose invalid, skip this frame
                            continue
                try:
                    # Get data
                    # Note: hand_tracker_data is already in the chest first frame coordinate system (left_eef_pose_in_chest)
                    hand_tracker_data = hdf5_tracker_data[tracker_id][tracker_timestamp]
                    chest_tracker_data = hdf5_tracker_data[self.chest_tracker_id][chest_timestamp]
                    # Check if wrist2tracker transformation exists
                    wrist2tracker = self._get_wrist2tracker_from_hdf5(hdf5_file, is_left_hand)
                    if wrist2tracker is not None:
                        # wrist_existed=True: need to calculate wrist position
                        wrist_in_chest = hand_tracker_data @ wrist2tracker
                    else:
                        # wrist_existed=False: tracker is wrist, use hand_tracker_data directly
                        wrist_in_chest = hand_tracker_data

                    # Correct implementation referencing under-wrist camera: transform wrist from chest first frame coordinate system to world coordinate system
                    wrist_pose_world = np.linalg.inv(chest_tracker_data) @ wrist_in_chest
                    # Extract wrist position in world coordinate system
                    wrist_position_world = wrist_pose_world[:3, 3]
                    wrist_position_world_homogeneous = np.append(wrist_position_world, 1.0)
                    # Use tracker2cam to transform points from world coordinate system to camera coordinate system
                    tracker2cam = np.linalg.inv(target_camera.cam2tracker)
                    wrist_position_cam_homogeneous = tracker2cam @ wrist_position_world_homogeneous
                    wrist_position_cam = wrist_position_cam_homogeneous[:3] / wrist_position_cam_homogeneous[3]

                    # Check if point is in front of camera
                    if wrist_position_cam[2] <= 0:
                        continue

                    # Project wrist position
                    wrist_3d = wrist_position_cam.reshape(1, 3)
                    wrist_2d, wrist_validity = self._project_to_image(
                        wrist_3d, target_camera.intrinsic, target_camera.distortion, 
                        target_camera.camera_model
                    )
                    # print('wrist_2d', wrist_2d)
                    # print('wrist_validity', wrist_validity)

                    if wrist_validity[0]:
                        # Calculate transparency (current frame brightest, future frames gradually darken)
                        distance_from_current = index - current_index  # Future distance (>=0)
                        # Current frame alpha = 1.0, future frames gradually decrease to 0.3
                        alpha = max(0.3, 1.0 - distance_from_current / self.trajectory_range * 0.7)

                        wrist_points.append({
                            'position': wrist_2d[0],
                            'index': index,
                            'is_current': index == current_index,
                            'alpha': alpha,
                            'is_future': index > current_index
                        })

                except Exception as e:
                    continue

            # Draw wrist trajectory
            self._render_simple_wrist_trajectory(image, wrist_points)

        return image
    

if __name__ == "__main__":
    wiyh = WIYH(dataroot='/mnt/data/data/zyp/wiyh/data')
    # camera_list = ['ldl_hand_fisheye', 'ldr_hand_fisheye', 'lf_chest_fisheye', 'rdl_hand_fisheye', 'rdr_hand_fisheye', 'rf_chest_fisheye']
    # wiyh.show_all_camera_rgb("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000", camera_list)
    # camera = 'ldl_hand_fisheye'
    # anno = wiyh.visualize_depth("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    # wiyh.generate_glove_visualization("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000", 100)
    # wiyh.get_eps_meta("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    # wiyh.get_scene_meta("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1")
    # wiyh.list_eps("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1")
    # wiyh.vis_h5_structure("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    # wiyh.get_annoatations("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000", 'task_status')