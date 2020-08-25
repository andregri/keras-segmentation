import numpy as np


def mask_to_oneclass(mask_batch, train_id):
    """Apply a preprocessing to the mask: convert a label segmentation batch of
    masks to a onehot encoded batch of masks keeping the only the class
    specified by train_id.

    Arguments:
        mask_batch (numpy array): (batch_size, width, height). Batch of mask
            images.

        train_id (int): classes id to keep.

    Returns:
        onehot_batch (numpy array): (batch_size, width, height, len(class_ids))
            batch of masks onehot encoded where only the classes specified by
            the argument class_ids were kept.
    """
    if len(mask_batch.shape) != 3:
        raise ValueError("mask_batch must have a shape (batch_size, W, H)")

    mask_batch[mask_batch != train_id] = 0

    mask_batch /= train_id  # normalize
    # onehot_batch = to_categorical(mask_batch)

    onehot_batch = np.expand_dims(mask_batch, axis=-1)

    return onehot_batch


def my_generator(img_gen, mask_gen):
    gen = zip(img_gen, mask_gen)
    for (img, mask) in gen:
        mask = mask[:, :, :, 0:1]
        # mask = mask_to_oneclass(mask, train_id)
        yield (img, mask)
