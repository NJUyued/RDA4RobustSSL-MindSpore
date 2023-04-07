import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
import random
from mindspore.dataset import MnistDataset, GeneratorDataset
import numpy as np

from mindspore import dtype as mstype
from mindspore import nn

import torchvision
from torchvision import datasets


def split_ssl_data(data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.

    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx = sample_labeled_data(data, target, num_labels, num_classes, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(data, target,
                        num_labels,
                        num_classes,
                        index=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''

    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)

# Iterable object as input source
class Iterable:
    def __init__(self,image, label,train):
        self._data = image
        self._label = label
        # self._data = np.random.sample((5, 2))
        # self._label = np.random.sample((5, 1))
        self.train = train

    def __getitem__(self, index):
        if not self.train:
            return self._data[index], self._label[index]
        else:
            return self._data[index], self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)

def randaugment(num_augmentations=2, magnitude=9):
    augmentations = [
        C.RandomRotation(degrees=(-magnitude, magnitude)),
        C.RandomColor(magnitude),
        C.RandomSharpness(magnitude),
        C.RandomAutoContrast(),
        C.RandomSolarize(magnitude),
        C.RandomInvert(),
        C.RandomEqualize(),
        C.RandomShearX(magnitude),
        C.RandomShearY(magnitude),
        C.RandomTranslateX(magnitude),
        C.RandomTranslateY(magnitude)
    ]

    chosen_augmentations = random.sample(augmentations, num_augmentations)
    return chosen_augmentations

def create_dataset(dataset, num_labels, num_classes, batch_size):
    lb_dst, ulb_dst, val_dst = None, None, None

    trans = []
    trans += [
        vision.Resize(32),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]
    w_trans = trans + [
        #vision.RandomCrop((32, 32), (4, 4, 4, 4)),
        vision.RandomHorizontalFlip(prob=0.5)
    ]
    augmentation_ops = randaugment(5, 9)
    s_trans = trans + augmentation_ops
    target_trans = transforms.TypeCast(mstype.int32)

    dset = getattr(torchvision.datasets, dataset.upper())
    dset = dset('../data', train=True, download=False)
    data, targets = dset.data, dset.targets

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets,
                                                                num_labels, num_classes,
                                                                None, False)

    sampler = ds.RandomSampler(replacement=True, num_samples = 1000)

    lb_data = Iterable(lb_data, lb_targets, True)
    lb_dst = GeneratorDataset(source=lb_data, column_names=["w_data", "s_data", "label"], sampler=sampler)

    ulb_data = Iterable(ulb_data, ulb_targets, True)
    ulb_dst = GeneratorDataset(source=ulb_data, column_names=["w_data", "s_data","label"], sampler=sampler)

    lb_dst = lb_dst.map(operations=w_trans,input_columns='w_data')
    lb_dst = lb_dst.map(operations=s_trans, input_columns='s_data')
    lb_dst = lb_dst.map(operations=target_trans, input_columns='label')
    lb_dst = lb_dst.batch(batch_size)


    ulb_dst = ulb_dst.map(operations=w_trans, input_columns='w_data')
    ulb_dst = ulb_dst.map(operations=s_trans, input_columns='s_data')
    ulb_dst = ulb_dst.map(operations=target_trans, input_columns='label')
    ulb_dst = ulb_dst.batch(batch_size)

    dset = getattr(torchvision.datasets, dataset.upper())
    dset = dset('../data', train=False, download=False)
    data, targets = dset.data, dset.targets

    val_data = Iterable(data, targets, False)
    val_dst = GeneratorDataset(source=val_data, column_names=["data", "label"])

    val_dst = val_dst.map(operations=trans, input_columns='data')
    val_dst = val_dst.map(operations=target_trans, input_columns='label')
    val_dst = val_dst.batch(batch_size)

    return lb_dst, ulb_dst, val_dst

if __name__ == '__main__':
    _,_,dst = create_dataset('cifar10',40,10,64)
    val_loader = dst.create_tuple_iterator()
    print(len(dst))

# data = Iterable()
# dataset = GeneratorDataset(source=data, column_names=["data", "label"])

