import os
import tarfile

import scipy.io

if __name__ == "__main__":
    imagenet_valid_tar_path = "./dataset/imagenet/val_images.tar.gz"
    target_dir = "./dataset/imagenet/val"  
    meta_path = "./dataset/imagenet/ILSVRC2012_devkit_t12/data/meta.mat"
    trueth_label_path = "./dataset/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

    meta = scipy.io.loadmat(meta_path, squeeze_me=True)
    ilsvrc2012_id_to_wnid = {m[0]: m[1] for m in meta["synsets"]}

    with open(trueth_label_path, "r") as f:
        ilsvrc_ids = tuple(int(ilsvrc_id) for ilsvrc_id in f.read().split("\n")[:-1])

    for ilsvrc_id in ilsvrc_ids:
        wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
        os.makedirs(os.path.join(target_dir, wnid), exist_ok=True)

    os.makedirs(target_dir, exist_ok=True)
    num_valid_images = 50000
    import shutil

    with tarfile.open(imagenet_valid_tar_path, mode="r") as tar:
        for valid_id, ilsvrc_id in zip(range(1, num_valid_images + 1), ilsvrc_ids):
            wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
            filename = "ILSVRC2012_val_{}_{}.JPEG".format(str(valid_id).zfill(8), str(wnid))
            from_ = f"temp/{filename}"
            to_ = f"{target_dir}/{wnid}/{filename}"
            print(from_, "==>", to_)
            shutil.move(from_, to_)
