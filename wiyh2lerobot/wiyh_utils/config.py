WIYH_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 320, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 320, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "w", "x", "y", "z"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z","w", "x", "y", "z"]},
    },
    "instruction": {
        "dtype": "string",
        "shape": (1,),
        "names": ["instruction"],
    },
}
