import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path) # returns basename of the file
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[1]), resample=Image.BILINEAR
    )
    img.save(outpath)
