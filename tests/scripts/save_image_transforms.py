from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.common.utils.utils import init_hydra_config, seeded_context
from tests.test_transforms import ARTIFACT_DIR, REPO_ID
from tests.utils import DEFAULT_CONFIG_PATH


def save_default_config_transform(original_frame: torch.Tensor, output_dir: Path):
    cfg = init_hydra_config(DEFAULT_CONFIG_PATH)
    cfg_tf = cfg.training.image_transforms
    default_tf = get_image_transforms(
        brightness_weight=cfg_tf.brightness.weight,
        brightness_min_max=cfg_tf.brightness.min_max,
        contrast_weight=cfg_tf.contrast.weight,
        contrast_min_max=cfg_tf.contrast.min_max,
        saturation_weight=cfg_tf.saturation.weight,
        saturation_min_max=cfg_tf.saturation.min_max,
        hue_weight=cfg_tf.hue.weight,
        hue_min_max=cfg_tf.hue.min_max,
        sharpness_weight=cfg_tf.sharpness.weight,
        sharpness_min_max=cfg_tf.sharpness.min_max,
        max_num_transforms=cfg_tf.max_num_transforms,
        random_order=cfg_tf.random_order,
    )

    with seeded_context(1337):
        img_tf = default_tf(original_frame)

    save_file({"default": img_tf}, output_dir / "default_transforms.safetensors")


def save_single_transforms(original_frame: torch.Tensor, output_dir: Path):
    transforms = [
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "sharpness",
    ]

    frames = {"original_frame": original_frame}
    for transform in transforms:
        kwargs = {
            f"{transform}_weight": 1.0,
            f"{transform}_min_max": (0.5, 0.5),
        }
        tf = get_image_transforms(**kwargs)
        frames[transform] = tf(original_frame)

    save_file(frames, output_dir / "single_transforms.safetensors")


def main():
    dataset = LeRobotDataset(REPO_ID, image_transforms=None)
    output_dir = Path(ARTIFACT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_frame = dataset[0][dataset.camera_keys[0]]

    save_single_transforms(original_frame, output_dir)
    save_default_config_transform(original_frame, output_dir)


if __name__ == "__main__":
    main()
