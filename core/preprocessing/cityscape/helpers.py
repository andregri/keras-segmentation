import numpy as np


def color_gt_image(gt_img, id2color):
    """Take a ground-truth image where each pixel's value corresponds to a class
    ID (1-channel image) and return a RGB image (3-channels) where each class
    has a different color. label2color maps from class to color.

    Arguments:
    gt_img (numpy array): ground-truth 1-channel image where each pixel's value
        corresponds to a class ID. Shape=(width,height).
    id2color (dict): dictionary whose keys are class IDs and values are
        triplets of ints (RGB color).

    Returns:
    color_img (numpy array): ground-truth 3-channels image where each colored
        pixel corresponds to a class ID. Shape=(width,height,3).
    """

    color_img = np.zeros(shape=(gt_img.shape[0], gt_img.shape[1], 3))
    for id, color in id2color.items():
        color_img[gt_img == id] = color

    return color_img

def IoU(true_label_mask, pred_label_mask, id2name):
    """Mean Intersection over Union = TP / (FN + TP + FP)
    Given the ground truth images and the predicted masks, compute the mIoU.

    Arguments:
        true_label_mask (np array): (width, height, n_classes) ground-truth
            masks.
        pred_label_mask (np array): (width, height, n_classes) predicted masks.
        id2name (dict): a dictionary whose keys are class IDs and values are the
            class names.
    """
    IoUs = []
    for i in id2name.keys():
        TP = np.sum((true_label_mask == i) & (pred_label_mask == i))
        FP = np.sum((true_label_mask != i) & (pred_label_mask == i))
        FN = np.sum((true_label_mask == i) & (pred_label_mask != i))
        IoU = TP / float(FN + TP + FP)
        IoUs.append(IoU)
        print(f"{id2name[i]:22}:\t#TP={TP:6},\t#FP={FP:6},\t#FN={FN:6},\tIoU={IoU:4.3f}")

    mIoU = np.mean(IoUs)
    print(f"mean IoU: {mIoU:4.3f}")
