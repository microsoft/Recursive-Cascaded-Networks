
import json
import copy
import numpy as np
import collections

from .liver import FileManager
from .liver import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, split_path, paired=False, task=None, batch_size=None):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            for entry in v:
                self.subset[k][entry] = self.files[entry]

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]

        self.image_size = config.get("image_size", [128, 128, 128])
        self.segmentation_class_value = config.get(
            'segmentation_class_value', None)

        if 'atlas' in config:
            self.atlas = self.files[config['atlas']]
        else:
            self.atlas = None

        self.batch_size = batch_size

    def center_crop(self, volume):
        slices = [slice((os - ts) // 2, (os - ts) // 2 + ts) if ts < os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        volume = volume[slices]

        ret = np.zeros(self.image_size, dtype=volume.dtype)
        slices = [slice((ts - os) // 2, (ts - os) // 2 + os) if ts > os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        ret[slices] = volume

        return ret

    @staticmethod
    def generate_atlas(atlas, sets, loop=False):
        sets = copy.copy(sets)
        while True:
            if loop:
                np.random.shuffle(sets)
            for d in sets:
                yield atlas, d
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False):
        if batch_size is None:
            batch_size = self.batch_size
        scheme = self.schemes[subset]
        if 'registration' in self.task:
            if self.atlas is not None:
                generators, fractions = zip(*[(self.generate_atlas(self.atlas, list(
                    self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])
            else:
                generators, fractions = zip(
                    *[(self.generate_pairs(list(self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])

            while True:
                imgs = [batch_size] + self.image_size + [1]
                ret = dict()
                ret['voxel1'] = np.zeros(imgs, dtype=np.float32)
                ret['voxel2'] = np.zeros(imgs, dtype=np.float32)
                ret['seg1'] = np.zeros(imgs, dtype=np.float32)
                ret['seg2'] = np.zeros(imgs, dtype=np.float32)
                ret['point1'] = np.ones(
                    (batch_size, 6, 3), dtype=np.float32) * (-1)
                ret['point2'] = np.ones(
                    (batch_size, 6, 3), dtype=np.float32) * (-1)
                ret['id1'] = np.empty((batch_size), dtype='<U30')
                ret['id2'] = np.empty((batch_size), dtype='<U30')

                i = 0
                flag = True
                cc = collections.Counter(np.random.choice(range(len(fractions)), size=[
                                         batch_size, ], replace=True, p=fractions))
                nums = [cc[i] for i in range(len(fractions))]
                for gen, num in zip(generators, nums):
                    assert not self.paired or num % 2 == 0
                    for t in range(num):
                        try:
                            while True:
                                d1, d2 = next(gen)
                                break
                        except StopIteration:
                            flag = False
                            break

                        if 'segmentation' in d1:
                            ret['seg1'][i, ..., 0] = d1['segmentation']
                        if 'segmentation' in d2:
                            ret['seg2'][i, ..., 0] = d2['segmentation']

                        ret['voxel1'][i, ..., 0], ret['voxel2'][i, ...,
                                                                0] = d1['volume'], d2['volume']
                        ret['id1'][i] = d1['id']
                        ret['id2'][i] = d2['id']
                        i += 1

                if flag:
                    assert i == batch_size
                    yield ret
                else:
                    yield ret
                    break
