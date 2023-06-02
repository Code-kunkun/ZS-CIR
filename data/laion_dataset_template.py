from torch.utils.data import Dataset
import json 
import PIL 
from PIL import Image 
from PIL import ImageFile
import os 
data_file_path = os.path.dirname(__file__)
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True 


class LaionDataset_Template(Dataset):
    def __init__(self, split: str, preprocess: callable):
        self.preprocess = preprocess
        self.split = split

        if split not in ['train']:
            raise ValueError("split should be in ['train']")

        self.image_path_prefix = "/GPFS/public/laion_coco_metadata_600m/images/"
        with open(data_file_path + "/files/laion_template_info.json") as f:
            self.triplets = json.load(f)

        print(f"Laion {split} dataset initialized")

    def __getitem__(self, index):

        reference_image = f"{str(self.triplets[index]['ref_image_id']).zfill(7)}.png"
        relative_caption = self.triplets[index]['relative_cap']
        target_image = f"{str(self.triplets[index]['tgt_image_id']).zfill(7)}.png"

        reference_image_path = self.image_path_prefix + reference_image 
        reference_image = PIL.Image.open(reference_image_path)
        if reference_image.mode == 'RGB':
            reference_image = reference_image.convert('RGB')
        else:
            reference_image = reference_image.convert('RGBA')
        reference_image = self.preprocess(reference_image)
        target_image_path = self.image_path_prefix + target_image
        target_image = PIL.Image.open(target_image_path)
        if target_image.mode == 'RGB':
            target_image = target_image.convert('RGB')
        else:
            target_image = target_image.convert('RGBA')
        target_image = self.preprocess(target_image)
        return reference_image, target_image, relative_caption

    def __len__(self):
        return len(self.triplets)
