import math
import cv2
import dask.array as da
from torch.utils.data import Dataset
from empanada.array_utils import take

def resize(image, scale_factor=1):
    # do nothing
    if scale_factor == 1:
        return image

    # cv2 expects (w, h) for image size
    h, w = image.shape
    dh = math.ceil(h / scale_factor)
    dw = math.ceil(w / scale_factor)

    image = cv2.resize(image, (dw, dh), cv2.INTER_LINEAR)

    return image

class VolumeDataset(Dataset):
    def __init__(self, array, axis=0, tfs=None, scale=1):
        super(VolumeDataset, self).__init__()
        if not math.log(scale, 2).is_integer():
            raise Exception(f'Image rescaling must be log base 2, got {scale}')

        self.array = array
        self.axis = axis
        self.tfs = tfs
        self.scale = scale

    def __len__(self):
        return self.array.shape[self.axis]

    def __getitem__(self, idx):
        # load the image
        image = take(self.array, idx, self.axis)

        # if dask, then call compute
        if type(image) == da.core.Array:
            image = image.compute()

        # downsample image by scale
        image = resize(image, self.scale)
        image = self.tfs(image=image)['image']

        return {'index': idx, 'image': image}
