from glob import glob
from pathlib import Path, PurePath
from enum import Enum, auto
import cv2
import torch
import torchvision
import numpy as np

DATA_DIR_PATH = PurePath("data/DFire/")


class Class(Enum):
    SMOKE = 0
    FIRE = 1


def get_smoky_and_non_smokey_imgs(root_dir_path: str):
    all_img_paths = set()
    smokey_img_paths = set()
    firey_img_paths = set()
    for label_path in glob(
        str(PurePath(root_dir_path) / "**" / "*.txt"), recursive=True
    ):
        with open(label_path) as file:
            lines = file.readlines()
        classes = {int(line.rstrip().split()[0]) for line in lines}

        img_id = PurePath(label_path).stem
        img_name = PurePath(img_id).with_suffix(".jpg").name
        img_path = str(PurePath(label_path).parent.parent / "images" / img_name)

        all_img_paths.add(img_path)

        if Class.FIRE.value in classes:
            firey_img_paths.add(img_path)

        if Class.SMOKE.value in classes:
            smokey_img_paths.add(img_path)

    only_smoke_img_paths = smokey_img_paths.difference(firey_img_paths)
    empty_img_paths = all_img_paths.difference(smokey_img_paths).difference(
        firey_img_paths
    )
    return only_smoke_img_paths, empty_img_paths


def clean_and_copy(save_dir_path: str, img_paths: list[str]):
    Path(save_dir_path).mkdir(exist_ok=True)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = img.transpose((2, 0, 1))
        min_img_dim = min(img.shape[1], img.shape[2])

        img = torch.tensor(img)
        img = torchvision.transforms.CenterCrop((min_img_dim, min_img_dim))(img)
        img = np.array(img)
        img = img.transpose((1, 2, 0))
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        save_img_path = str(PurePath(save_dir_path) / PurePath(img_path).name)
        cv2.imwrite(save_img_path, img)


def main():
    train_only_smoke_img_paths, train_empty_img_paths = get_smoky_and_non_smokey_imgs(
        str(PurePath(DATA_DIR_PATH) / "raw" / "train")
    )
    test_only_smoke_img_paths, test_empty_img_paths = get_smoky_and_non_smokey_imgs(
        str(PurePath(DATA_DIR_PATH) / "raw" / "test")
    )
    clean_and_copy(
        str(PurePath(DATA_DIR_PATH) / "clean" / "train" / "smoke"),
        train_only_smoke_img_paths,
    )
    clean_and_copy(
        str(PurePath(DATA_DIR_PATH) / "clean" / "train" / "empty"),
        train_empty_img_paths,
    )
    clean_and_copy(
        str(PurePath(DATA_DIR_PATH) / "clean" / "test" / "smoke"),
        test_only_smoke_img_paths,
    )
    clean_and_copy(
        str(PurePath(DATA_DIR_PATH) / "clean" / "test" / "empty"),
        test_empty_img_paths,
    )


if __name__ == "__main__":
    main()
