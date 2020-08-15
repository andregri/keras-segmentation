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
        if img_path.stem.replace("_leftImg8bit", "") is \
                label_path.stem.replace("_gtFine_labelIds", ""):
            pair = (img_path, label_path)
            pairs.append(pair)

    return pairs
