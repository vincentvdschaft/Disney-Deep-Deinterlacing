from pathlib import Path
import torchvision
import cv2
import torch
from src.utils.common import *
from .datasets import Dataset, DatasetConfig, register_dataset


class Vimeo90kTripletDatasetConfig(DatasetConfig):
    """Config for Vimeo090kTripletDataset."""

    def _validate(self):
        super()._validate()
        self._must_contain('dataset_root', element_type=str)
        self._must_contain('resize', (bool, list, tuple))
        self._must_contain('num_color_channels', int, in_set=(1, 3))


@register_dataset(dataset_tag='vimeo_triplet', dataset_config_class=Vimeo90kTripletDatasetConfig)
class Vimeo090kTripletsDataset(Dataset):
    """Dataset of video frame field triplets"""

    def __init__(self, config, train=False, transform=None, *args, **kwargs):
        Dataset.__init__(self, config)

        if transform is None:
            transform = []
        else:
            transform = [transform]

        if isinstance(self.config.resize, (tuple, list)):
            transform.append(torchvision.transforms.Resize(
                self.config.resize))

        self.transform = torchvision.transforms.Compose(
            transform
        )

        if train:
            string = 'train'
        else:
            string = 'test'

        sample_location_index_path = Path(self.config.dataset_root,
                                          f'tri_{string}list.txt')

        with open(sample_location_index_path, 'r', encoding='utf-8') as f:
            self.folder_paths = f.read().splitlines()

        # Remove missing samples from list
        root = self.config.dataset_root
        self.folder_paths = list(filter(
            lambda p: Path(root, 'sequences', p).is_dir() and len(p) > 0,
            self.folder_paths
        ))

    def __getitem__(self, index: int):
        """Returns a sample from the dataset which returns the known fields as
        well as the estimation target in the form of ((I1+, I2-, I3+),I2+),
        where a plus indicates the - indicates the odd columns and the +
        indicates the even columns."""

        folder_path = Path(self.folder_paths[index])

        imgs = []

        for n in range(3):
            path = str(Path(
                self.config.dataset_root, 'sequences',
                folder_path, f'im{n+1}.png'))
            # Load the image
            if self.config.num_color_channels == 1:
                im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                # Add color channel dimension
                im = im[None]
            else:
                im = cv2.imread(path, cv2.IMREAD_COLOR)
                # Fix colors (opencv uses BGR while matplotlib uses RGB)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # Move color channel dimension to front
                im = np.transpose(im, (2, 0, 1))

            assert im is not None, f'image from {folder_path} could not be loaded!'
            
            # Turn into tensor
            im = torch.tensor(im, dtype=torch.float32)/255

            # Add batch dimension
            im = im[None]
            # Apply transformations
            im = self.transform(im)
            # im has shape (B, C, H, W)
            # The elements of imgs have shape (C, H, W)
            imgs.append(im[0])

        # Get the known fields (I1+, I2-, I3+)
        I_1_plus = imgs[0][:, :, 0::2]  # shape: (C, H, W)
        I_2_min = imgs[1][:, :, 1::2]  # shape: (C, H, W)
        I_3_plus = imgs[2][:, :, 0::2]  # shape: (C, H, W)

        # Get the even columns of the middle frame I2+ (the target to estimate)
        I_2_plus = imgs[1][:, :, 0::2]  # shape: (C, H, W)

        # Concatenate along channel dimension
        known_fields = torch.concat([I_1_plus, I_2_min, I_3_plus], dim=0)

        # Shape: (C, H, W)
        return known_fields, I_2_plus

    def __len__(self) -> int:
        return len(self.folder_paths)

    def _field_to_im(self, field, even=True):
        """Turns a field tensor where half the lines are missing into a tensor
        where the missing lines are present and all ones.

        Args:
            field (:obj:`Tensor` (H, W)): The field tensor
            even: Set to True if the field contains the even lines

        Returns:
            tensor of shape (H, 2xW)
        """
        H, W = field.shape

        im = torch.ones((H, 2*W))
        if even:
            im[:, ::2] = field
        else:
            im[:, 1::2] = field

        return im
