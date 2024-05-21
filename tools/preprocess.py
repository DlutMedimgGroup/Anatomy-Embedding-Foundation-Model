from multiprocessing.pool import ThreadPool
import numpy as np
import pickle
import torch
from typing import Dict, Hashable, Mapping
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    MapTransform,
    ScaleIntensityRange
)
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from utility import *

class amosScaleIntensityRanged(MapTransform):

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange(a_min=-500, a_max=750, b_min=-1, b_max=1, clip=True, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d

class LabelTyped(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        values, _ = torch.unique(d['image'], return_inverse=True)
        d['image_meta_dict']['labels'] = list(values.numpy().astype(np.int32))
        return d

def generate(file_names):
    image_or_label, in_file_name, out_file_name = file_names

    if image_or_label == 'image':
        transform = Compose(
                [
                    LoadImaged(keys=["image"], image_only=False),
                    EnsureChannelFirstd(keys=["image"]),
                    EnsureTyped(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="LPS"),
                    amosScaleIntensityRanged(keys="image"),
                    Spacingd(
                        keys=["image"],
                        pixdim=[1.0, 1.0, 1.0],
                        mode=("bilinear"),
                    ),
                ]
        )
    elif image_or_label == 'label':
        transform = Compose(
                [
                    LoadImaged(keys=["image"], image_only=False),
                    EnsureChannelFirstd(keys=["image"]),
                    EnsureTyped(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="LPS"),
                    LabelTyped(keys="image"),
                    Spacingd(
                        keys=["image"],
                        pixdim=[1.0, 1.0, 1.0],
                        mode=("nearest"),
                    ),
                    
                ]
        )

    output = transform({'image': in_file_name})
    array = output['image'].numpy()
    if np.max(array) < 0.1:
        return
    dict = output['image_meta_dict']
    with open(out_file_name[:-4]+'.pkl', 'wb') as f:
        pickle.dump(dict, f)
    np.save(out_file_name, array)
    basename = os.path.basename(out_file_name)
    print(basename + ' finished.')


def main():
    # in_path = '/home/mingrui/disk1/projects/monai/Projects/202310_Geodesic/source_data/Scribble611_BraTS'
    # out_path = '/mnt/nfs_ssd/mingrui/dataset/Scribble611_BraTS'
    # folders = [
    #     # ['image or label', 'in_folder_name', 'out_folder_name', 'prefix', 'suffix',]
    #     ['image', 'imagesTr', 'imagesTr', None, None],
    #     ['label', 'labelsTr', 'labelsTr', None, None],
    #     ['label', 'labelsTr', 'scribblesTr', None, None],
    # ]
    # ScaleIntensityRange = [1, 0]

    in_path = '/home/mingrui/disk1/projects/monai/Projects/202403_SAM_improve/source_data/moose_5'
    out_path = '/mnt/nfs_ssd/mingrui/dataset/moose_5'
    folders = [
        # ['image or label', 'in_folder_name', 'out_folder_name', 'prefix', 'suffix',]
        ['image', 'imagesTr', 'imagesTr', None, None],
        ['label', 'labelsTr', 'labelsTr', None, None],
    ]

    tasks = []
    for folder in folders:
        image_or_label, in_folder_name, out_folder_name, prefix, suffix = folder
        data_list = subfiles(join(in_path, in_folder_name), prefix=prefix, suffix=suffix, join=False)
        print('number of data to convert: ' + str(len(data_list)))
        maybe_mkdir_p(join(out_path, out_folder_name))
        
        for data_name in data_list:
            in_file_name = join(in_path, in_folder_name, data_name)
            out_file_name = join(out_path, out_folder_name, data_name[:-7]+'.npy')
            tasks.append((image_or_label, in_file_name, out_file_name))
    
    pool_size = 16
    pool = ThreadPool(pool_size)  # 创建一个线程池
    pool.map(generate, tasks)  # 往线程池中填线程
    pool.close()  # 关闭线程池，不再接受线程
    pool.join()  # 等待线程池中线程全部执行完

    
    

if __name__ == "__main__":
    main()
