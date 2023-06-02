from torch.utils.data import Dataset
import json 
import PIL 
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 2300000000


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        if self.split == 'test_train':
            split = 'train'

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val'")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(f'/GPFS/data/yikunliu/CIRR/cirr/captions/cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(f'/GPFS/data/yikunliu/CIRR/cirr/image_splits/split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption'].lower()

                if self.split == 'train':
                    reference_image_path = "/GPFS/data/yikunliu/NLVR2/images/" + self.name_to_relpath[reference_name][2:]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = "/GPFS/data/yikunliu/NLVR2/images/" + self.name_to_relpath[target_hard_name][2:]
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert("RGB"))
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    reference_image_path = "/GPFS/data/yikunliu/NLVR2/images/" + self.name_to_relpath[reference_name][2:]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members, reference_image

                elif self.split == 'test1':
                    reference_image_path = "/GPFS/data/yikunliu/NLVR2/images/" + self.name_to_relpath[reference_name][2:]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members, reference_image 

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = "/GPFS/data/yikunliu/NLVR2/images/" + self.name_to_relpath[image_name][2:]
                im = PIL.Image.open(image_path).convert("RGB")
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
