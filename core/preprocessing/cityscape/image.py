import os
import sys
from pathlib import Path

from tqdm import trange
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    save_img
)
from matplotlib import pyplot as plt
import numpy as np


def keep_trainId_gt(cityscapes_dir: Path) -> None:
    """Take the path to the cityscapes dataset and remove the ground-truth
    images that don't contain "labelIds" in their name.

    Arguments:
        cityscapes_dir (Path): path to the cityscapes directory (the directory
        that contains "gtFine_trainvaltest" e "leftImg8bit_trainvaltest").
    """
    gt_dir = cityscapes_dir / "gtFine_trainvaltest" / "gtFine"
    if not gt_dir.is_dir():
        raise ValueError("[-] The path provided is not"
                         " a cityscapes directory.")

    print("[+] Remove useless gt images...")
    tot = count_files(gt_dir, "png")
    print(f"[+] Number of gt images: {tot}")

    mask_path_gen = (path for path in gt_dir.rglob("*.png"))

    removed = 0
    for _ in trange(tot):
        p = next(mask_path_gen)
        if "TrainIds" not in p.stem:
            os.remove(p)
            removed += 1

    print(f"[+] Number of removed gt images: {removed}")
    print(f"[+] Number of remaining gt images: {tot-removed}")


def resize(cityscapes_dir: Path, target_shape: tuple) -> None:
    """Take the path to the cityscapes dataset and resize all images to the
    target shape.

    Arguments:
        cityscapes_dir (Path): path to the cityscapes directory (the directory
        that contains "gtFine_trainvaltest" e "leftImg8bit_trainvaltest").

        target_shape (tuple): (width, height) with the desired shape.
    """
    if not cityscapes_dir.is_dir():
        raise ValueError("The path provided is not a cityscapes directory.")

    if len(target_shape) != 2:
        raise ValueError("Target shape must be a tuple (width, height).")

    print("[+] Resizing all the images...")
    tot = count_files(cityscapes_dir, "png")
    print(f"[+] Number of images: {tot}")

    mask_path_gen = (path for path in cityscapes_dir.rglob("*.png"))

    for _ in trange(tot):
        p = next(mask_path_gen)
        img = img_to_array(load_img(p, target_size=target_shape))
        save_img(
            p,
            img,
            data_format="channels_last",
            file_format="png",
            scale=False
        )


def remove_empty_imgs(cityscapes_dir: Path) -> None:
    """This function must be called after "keep_one_class". If the mask is
    empty, that is it has no pixels to 1, then remove this mask and the related
    RGB image.

    Arguments:
        cityscapes_dir (Path): path to the cityscapes directory (the directory
        that contains "gtFine_trainvaltest" e "leftImg8bit_trainvaltest").
    """
    gt_dir = cityscapes_dir / "gtFine_trainvaltest" / "gtFine"
    if not gt_dir.is_dir():
        raise ValueError("The path provided is not a cityscapes directory.")

    print("[+] Remove empty images...")
    tot = count_files(gt_dir, "png")
    print(f"[+] Number of gt images: {tot}")

    pair_gen = pair_generator(cityscapes_dir)
    removed = 0
    for _ in trange(tot):
        (img_path, mask_path) = next(pair_gen)
        mask = img_to_array(load_img(mask_path))  # (h,w,3)
        mask = mask[:, :, 0].astype(int)

        all_zeros = not np.any(mask)
        if all_zeros:
            os.remove(img_path)
            os.remove(mask_path)
            removed += 1

    print(f"Number of removed images: {removed}")
    print(f"Number of remaining images: {tot-removed}")


def count_files(dir: Path, file_format: str) -> int:
    path_gen = (path for path in dir.rglob("*."+file_format))

    tot = 0
    for p in path_gen:
        if not p.is_dir():
            tot += 1

    return tot


def pair_generator(cityscapes_dir: Path) -> tuple:
    """Generates a pair made of an image path and the corresponding mask path.

    Arguments:
        cityscapes_dir (Path): path to the cityscapes directory (the directory
        that contains "gtFine_trainvaltest" e "leftImg8bit_trainvaltest").

    Yields:
        pair (tuple): the image path and the corresponding mask path.
    """
    dir = cityscapes_dir
    if not dir.is_dir():
        raise ValueError("The path provided is not a cityscapes directory.")

    gt_dir = cityscapes_dir / "gtFine_trainvaltest" / "gtFine"
    img_dir = cityscapes_dir / "leftImg8bit_trainvaltest" / "leftImg8bit"

    mask_path_gen = (path for path in gt_dir.rglob("*.png"))

    for mask_p in mask_path_gen:
        if not mask_p.is_dir():
            mask_abs_path = mask_p.as_posix()

            img_filename = mask_p.name.replace("_gtFine_labelTrainIds",
                                               "_leftImg8bit")
            trainvaltest = mask_p.parent.parent.stem
            city = mask_p.parent.stem

            img_p = img_dir / trainvaltest / city / img_filename
            if not img_p.exists():
                raise ValueError(f"The mask '{mask_abs_path}'"
                                 " has no a corresponding image."
                                 f"'{img_p}' is not valid")
            yield (img_p, mask_p)


def show_oneclass_pair(img_path: Path, mask_path: Path):
    """Display an image and the one class mask.
    """
    img = img_to_array(load_img(img_path))  # (h,w,3)
    original_mask = img_to_array(load_img(mask_path))  # (h,w,3)

    _, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs = axs.flatten()

    # Display the image
    axs[0].imshow(img / 255.0)
    axs[0].set_title(img_path.stem)
    axs[0].set_xlabel(img.shape)

    # Display the mask
    axs[1].imshow(original_mask / 255.0)
    axs[1].set_title(mask_path.stem)
    axs[1].set_xlabel(original_mask.shape)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: image.py cityscapes_root_dir")

    cityscapes_root_dir = sys.argv[1]
    path = Path(cityscapes_root_dir)
    target_shape = (640, 640)

    # Check that the number of images and masks matches:
    # for each image there are 3 png masks.
    imgs_path = path / "leftImg8bit_trainvaltest" / "leftImg8bit"
    n_imgs = count_files(imgs_path, "png")
    print(f"Number of images: {n_imgs}")
    masks_path = path / "gtFine_trainvaltest" / "gtFine"
    n_all_masks = count_files(masks_path, "png")
    print(f"Number of masks: {n_all_masks}")
    # assert n_imgs*4 == n_all_masks, "[-] Error, the number of images and masks don't match."

    # Remove the masks that are not labelIds and check that there are as many
    # masks as images.
    keep_trainId_gt(path)
    n_masks = count_files(masks_path, "png")
    assert n_imgs == n_masks, \
        "[-] Error, the number of images and masks don't match."

    # Show an image and its mask.
    pair_gen = pair_generator(path)
    (img_path, mask_path) = next(pair_gen)
    show_oneclass_pair(img_path, mask_path)

    # Keep only the traffic light class.
    remove_empty_imgs(path)
    new_n_imgs = count_files(imgs_path, "png")
    print(f"Number of images: {new_n_imgs}")
    new_n_masks = count_files(masks_path, "png")
    print(f"Number of masks: {new_n_masks}")

    # Show an image and its mask.
    pair_gen = pair_generator(path)
    (img_path, mask_path) = next(pair_gen)
    show_oneclass_pair(img_path, mask_path)

    # Resize all the images
    resize(path, target_shape)

    # Show an image and its mask.
    pair_gen = pair_generator(path)
    (img_path, mask_path) = next(pair_gen)
    show_oneclass_pair(img_path, mask_path)

    # Check that all images have the target shape
    for _ in trange(n_imgs):
        pair_gen = pair_generator(path)
        (img_path, mask_path) = next(pair_gen)
        img = img_to_array(load_img(img_path))  # (h,w,3)
        mask = img_to_array(load_img(mask_path))  # (h,w,3)
        assert img.shape[0:2] == target_shape
        assert mask.shape[0:2] == target_shape

    pair_gen = pair_generator(path)
    (img_path, mask_path) = next(pair_gen)
    show_oneclass_pair(img_path, mask_path)
