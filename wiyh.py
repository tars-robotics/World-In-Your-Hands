import os
import json
import numpy as np
import h5py
import sys
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Iterable, Optional, Dict
import laspy

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("wiyh dev-kit only supports Python version 3.")
MAX_DEPTH=2

class WIYH:
    
    def __init__(self,
                 version: str = 'v1.0-train',
                 dataroot: str = '/data/wiyh',
                 ):
        self.version = version
        self.dataroot = dataroot
        
        
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
    

if __name__ == "__main__":
    wiyh = WIYH(dataroot='/mnt/data/data/zyp/wiyh/data')
    # camera_list = ['ldl_hand_fisheye', 'ldr_hand_fisheye', 'lf_chest_fisheye', 'rdl_hand_fisheye', 'rdr_hand_fisheye', 'rf_chest_fisheye']
    # wiyh.show_all_camera_rgb("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000", camera_list)
    # camera = 'ldl_hand_fisheye'
    # anno = wiyh.visualize_depth("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    wiyh.get_eps_len("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    # wiyh.get_eps_meta("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    # wiyh.get_scene_meta("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1")
    # wiyh.list_eps("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1")
    # wiyh.vis_h5_structure("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000")
    # wiyh.get_annoatations("worldcode_HS-2-1422925000753_2025-11-03-13-24-37_1_s0_vlta_reorg_sample_1-1", "action_000", 'task_status')