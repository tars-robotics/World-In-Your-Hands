from pathlib import Path
from tqdm import tqdm
import numpy as np
from h5py import File


def load_local_episodes(input_h5: Path):
    with File(input_h5, "r") as f:
        for demo in tqdm(f["data"].values()):
            demo_len = len(demo["obs/lf_chest_rgb"])
            # (-1: open, 1: close) -> (0: close, 1: open)
            action = np.array(demo["left_actions"], dtype=np.float32)
            
            if len(action.shape) == 3:
                action = action[:, 0, :]
            else:
                continue
            episode = {
                "observation.images.image": np.array(demo["obs/lf_chest_rgb"]),
                "observation.images.wrist_image": np.array(demo["obs/eye_in_left_rgb"]),
                "observation.state": np.array(demo["left_states"], dtype=np.float32),
                "action": action,
                "instruction": [demo["instruction"][()].decode("utf-8")] * demo_len
            } 
            n1 = np.array(demo["obs/lf_chest_rgb"]).shape[0]
            n2 = np.array(demo["obs/eye_in_left_rgb"]).shape[0]
            n3 = np.array(demo["left_states"], dtype=np.float32).shape[0]
            n4 = action.shape[0]
            if n1 != demo_len or n2 != demo_len or n3 != demo_len or n4 != demo_len:
                continue
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]
