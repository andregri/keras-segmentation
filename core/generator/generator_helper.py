from tensorflow.keras.utils import to_categorical


def make_pair(img_paths, label_paths):
    """Take a list of image paths and the list of corresponding label paths and
    return a list of tuples, each containing one image path and the
    corresponding label path.

    Arguments:
      img_paths (list of `Path`): list of paths to images.
      labels_paths (list of `Path`): list of paths to labels.

    Returns:
      pairs (list of tuple): a list of tuples. Each tuple contain an image and
      the corresponding label.
    """
    if len(img_paths) != len(label_paths):
        raise ValueError("The lengths of the two lists mismatch.")

    pairs = []
    for img_path, label_path in zip(sorted(img_paths), sorted(label_paths)):
        img_stem = img_path.stem.replace("_leftImg8bit", "")
        label_stem = label_path.stem.replace("_gtFine_labelIds", "")
        if img_stem == label_stem:
            pair = (img_path, label_path)
            pairs.append(pair)

    return pairs


def my_generator(img_gen, mask_gen):
    gen = zip(img_gen, mask_gen)
    for (img, mask) in gen:
        mask = mask[:, :, :, 0]
        mask = to_categorical(mask, num_classes=35)
        yield (img, mask)
