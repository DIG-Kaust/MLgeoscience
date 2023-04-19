import os
import os.path

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class FossilNET(ImageFolder):
    """`FossilNET <https://github.com/softwareunderground/fossilnet>`_ Dataset.
    Modified from https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
    Args:
        root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
            and  ``EMNIST/processed/test.pt`` exist.
        split (string): The dataset has 3 different splits: ``train``, ``val``,
            ``test``. This argument specifies
            which one to use.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'https://swung-data.s3.amazonaws.com/fossilnet/fossilnet-png-224px.zip'
    folder = 'fossilnet-png-224px'
    md5 = '83e4f09fc78e3fd996c4e611c2653bf9'
    splits = ('train', 'val', 'test')

    def __init__(self, root, split, download=False, **kwargs):
        self.split = verify_str_arg(split, "split", self.splits)
        self.basedir = root
        os.makedirs(self.basedir, exist_ok=True)

        if self.split == "train":
            self.root = self.train_folder
        elif self.split == "val":
            self.root = self.val_folder
        elif self.split == "test":
            self.root = self.test_folder
        else:
            raise NotImplementedError

        if download:
            self.download()

        super(FossilNET, self).__init__(self.root, **kwargs)

    @property
    def train_folder(self):
        return os.path.join(self.basedir, self.folder, 'train')

    @property
    def val_folder(self):
        return os.path.join(self.basedir, self.folder, 'val')

    @property
    def test_folder(self):
        return os.path.join(self.basedir, self.folder, 'test')

    def _check_exists(self):
        return os.path.exists(self.root)

    def download(self):
        """Download the FossilNET data if it doesn't exist already."""
        if self._check_exists():
            return

        print('Downloading...')
        # download files
        download_and_extract_archive(self.url, download_root=self.basedir, filename=self.__class__.__name__+".zip", md5=self.md5)

        # process and save as torch files
        print('Done!')

    def extra_repr(self):
        return "Split: {}".format(self.split)


if __name__ == "__main__":
    from torchvision import transforms

    # From https://pytorch.org/hub/pytorch_vision_resnet/
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_normalize
    ])

    for phase in ["train", "val", "test"]:
        dataset = FossilNET("data", split=phase, download=True, transform=transform)
        X, y = dataset[0]

        print(dataset)
        print("Example Dataset Item: {0:}, {1:}".format(X.size(), dataset.classes[y]))
        print("")