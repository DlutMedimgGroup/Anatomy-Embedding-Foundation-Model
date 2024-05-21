from utility import *
import numpy as np
import random

def main_3group():
    # settings
    dataset_path = '/home/mingrui/disk1/projects/monai/Projects/202301_FedTS/source_data/BraTS_2018_Converted'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202301_FedTS/config_files/split_file_brats_small_fold1.json'

    number_of_train = 10
    number_of_val = 5
    offset = 0

    # working code
    training_files = []
    validation_files = []
    testing_files = []

    image_folder = join(dataset_path, 'images')
    label_folder = join(dataset_path, 'labels')

    image_list = subfiles(image_folder, join=False)
    image_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    number_of_data = len(image_list)

    print(f"number of data: {number_of_data}")

    for n, image_name in enumerate(image_list):
        if (n-offset) % int(number_of_data/(number_of_train + number_of_val)) == 0:
            if (len(training_files)/number_of_train) >= ((len(validation_files)+1)/number_of_val):
                validation_files.append(image_name)
            else:
                training_files.append(image_name)
        else:
            testing_files.append(image_name)
        
    output_dict = {'training data': [], 'validation data': [], 'testing data': []}

    for file_name in training_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['training data'].append({'image': image_name, 'label': label_name})

    for file_name in validation_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['validation data'].append({'image': image_name, 'label': label_name})
    
    for file_name in testing_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['testing data'].append({'image': image_name, 'label': label_name})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def main_4group():
    # settings
    dataset_path = '/home/mingrui/disk1/projects/monai/Projects/202301_FedTS/source_data/BraTS_2018_Converted'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202301_FedTS/config_files/split_file_brats_continue_learning.json'

    number_of_train = 10
    number_of_val = 5
    number_of_test = 57
    fold = 0
    number_of_fold = 5
    offset = 0

    # working code
    training_files = []
    validation_files = []
    addition_files = []
    testing_files = []

    image_folder = join(dataset_path, 'images')
    label_folder = join(dataset_path, 'labels')

    image_list = subfiles(image_folder, join=False)
    image_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    number_of_data = len(image_list)

    print(f"number of data: {number_of_data}")

    cache_list = []
    for n, image_name in enumerate(image_list):        
        if (n+number_of_fold-fold+1) % int(number_of_data/number_of_test) == 0:
            testing_files.append(image_name)
        else:
            cache_list.append(image_name)
    cache_id = np.array(list(range(number_of_train+number_of_val)))
    cache_id = cache_id*(len(cache_list)/(number_of_train+number_of_val))
    cache_id = np.floor(cache_id+offset)
    cache_id = list(cache_id)
    for n, image_name in enumerate(cache_list):
        if n in cache_id:
            if (len(training_files)/number_of_train) >= ((len(validation_files)+1)/number_of_val):
                validation_files.append(image_name)
            else:
                training_files.append(image_name)
        else:
            addition_files.append(image_name)
        
    output_dict = {'train': [], 'val': [], 'test': [], 'add': []}

    for file_name in training_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['train'].append({'image': image_name, 'label': label_name})

    for file_name in validation_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['val'].append({'image': image_name, 'label': label_name})
    
    for file_name in testing_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['test'].append({'image': image_name, 'label': label_name})

    for file_name in addition_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['add'].append({'image': image_name, 'label': label_name})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def main_4group_2():
    # settings
    dataset_path = '/home/mingrui/disk1/projects/monai/Projects/202301_FedTS/source_data/202302_FedSAM'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202301_FedTS/config_files/split_file_brats_continue_learning.json'

    image_folder = join(dataset_path, 'images')
    label_folder = join(dataset_path, 'labels')

    image_list = subfiles(image_folder, join=False)
    image_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")

    number_of_dataset1 = int(number_of_data/3)
    number_of_dataset2 = int(number_of_data/3)
    number_of_test = number_of_data - number_of_dataset1 - number_of_dataset2
    number_of_train1 = int(number_of_dataset1/5*4)
    number_of_val1 = number_of_dataset1 - number_of_train1
    number_of_train2 = int(number_of_dataset2/5*4)
    number_of_val2 = number_of_dataset2 - number_of_train2

    random.seed(1111)
    random.shuffle(image_list)

    pointer1 = 0
    pointer2 = pointer1+number_of_train1
    train1_files = image_list[pointer1:pointer2]
    train1_files.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    pointer1 = pointer2
    pointer2 = pointer1+number_of_val1
    val1_files = image_list[pointer1:pointer2]
    val1_files.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    pointer1 = pointer2
    pointer2 = pointer1+number_of_train2
    train2_files = image_list[pointer1:pointer2]
    train2_files.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    pointer1 = pointer2
    pointer2 = pointer1+number_of_val2
    val2_files = image_list[pointer1:pointer2]
    val2_files.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    pointer1 = pointer2
    pointer2 = pointer1+number_of_test
    test_files = image_list[pointer1:pointer2]
    test_files.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
        
    output_dict = {'train1': [], 'val1': [], 'train2': [], 'val2': [], 'test': []}

    for file_name in train1_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['train1'].append({'image': image_name, 'label': label_name})

    for file_name in val1_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['val1'].append({'image': image_name, 'label': label_name})
    
    for file_name in train2_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['train2'].append({'image': image_name, 'label': label_name})

    for file_name in val2_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['val2'].append({'image': image_name, 'label': label_name})

    for file_name in test_files:
        image_name = join(image_folder, file_name)
        label_name = join(label_folder, file_name)

        assert isfile(image_name), 'An image data does not exist. data name: '+image_name
        assert isfile(label_name), 'An label data does not exist. data name: '+label_name

        output_dict['test'].append({'image': image_name, 'label': label_name})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def _groups_helper(image_folder, groups, label_folder=None, prefix=None, suffix='.npy'):
    if label_folder == None:
        without_label = True
    else:
        without_label = False
    
    image_list = subfiles(image_folder, join=False, prefix=prefix, suffix=suffix)
    image_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")

    random.seed(1111)
    random.shuffle(image_list)

    output_dict = {}
    pointer1 = 0
    for key in groups:
        output_dict[key] = []
        pointer2 = pointer1 + groups[key]
        file_list = image_list[pointer1:pointer2]
        file_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
        pointer1 = pointer2
        groups[key] = file_list

    for key in output_dict:    
        for file_name in groups[key]:
            image_name = join(image_folder, file_name)
            assert isfile(image_name), 'An image data does not exist. data name: '+image_name
            if not without_label:
                if file_name[-9:-4] == "_0000":
                    file_name = file_name[:-9] + file_name[-4:]
                if file_name[-12:-7] == "_0000":
                    file_name = file_name[:-12] + file_name[-7:]
                label_name = join(label_folder, file_name)
                
                assert isfile(label_name), 'An label data does not exist. data name: '+label_name
                output_dict[key].append({'image': image_name, 'label': label_name})
            else:
                output_dict[key].append({'image': image_name,})
    return output_dict

def main_random_groups():
    # settings
    # dataset_path = '/home/mingrui/disk_ssd1/dataset/amos22_ct_mini_1mm'
    # split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/config_files/split_sam_amos22_ct_mini_1mm.json'
    # dataset_path = '/mnt/nfs_ssd/mingrui/dataset/flare_2023_1mm'
    # split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/config_files/split_sam_flare_1mm_seg.json'
    # dataset_path = '/mnt/nfs_ssd/mingrui/dataset/moose_5'
    # split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202403_SAM_improve/config_files/split_moose_5_1mm.json'
    dataset_path = '/home/mingrui/my_nas/Dataset/Flare2022/Testing'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202405_FLARE24/config_files/split_flare22_test.json'

    # image_folder = join(dataset_path, 'images')
    # label_folder = join(dataset_path, 'labels')
    image_folder = dataset_path
    groups = {
        'test': 200,
    }
    # outdict1 = _groups_helper(image_folder, groups, label_folder=label_folder, suffix='.nii.gz')
    outdict1 = _groups_helper(image_folder, groups, None, suffix='.nii.gz')
    # outdict1 = _groups_helper(image_folder, groups, prefix='FLARE', suffix='.npy')
    
    # dataset_path = '/mnt/nvme2n1/mingrui/dataset/flare_2023_1mm'
    # image_folder = join(dataset_path, 'unlabelTr1800')
    # groups = {
    #     'unlabelTr1800': 1800,
    # }
    # outdict2 = _groups_helper(image_folder, groups, prefix='FLARE', suffix='.npy')

    # dataset_path = '/home/mingrui/disk_ssd1/dataset/flare_2023_1mm'
    # image_folder = join(dataset_path, 'validation')
    # groups = {
    #     'validation': 100,
    # }
    # outdict3 = _groups_helper(image_folder, groups, prefix='FLARE', suffix='.npy')
    
    # output_dict = {**outdict1, **outdict2, **outdict3}
    output_dict = {**outdict1,}
    

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)



def sam_groups():
    # settings
    # dataset_path = '/home/mingrui/disk_ssd1/dataset/fed_dataset_amos22_resampled_1mm'
    dataset_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/source_data/amos22'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/config_files/split_sam_amos22.json'
    
    image_folder = join(dataset_path, 'imagesTr')
    groups = {
        'train': 240,
    }
    outdict1 = _groups_helper(image_folder, groups, prefix='amos', suffix='.nii.gz')
    
    image_folder = join(dataset_path, 'imagesVa')
    groups = {
        'val': 120,
    }
    outdict2 = _groups_helper(image_folder, groups, prefix='amos', suffix='.nii.gz')
    
    output_dict = {**outdict1, **outdict2}
    

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def brian_mri_groups():
    dataset_path = '/mnt/nfs_ssd/mingrui/dataset/brian_mri'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202402_MICCAI_brainmri/config_files/split_brain_mri_sam.json'

    image_folder = join(dataset_path, 'brain_norm')
    groups = {
        'brain_norm': 410,
    }
    outdict1 = _groups_helper(image_folder, groups, prefix=None, suffix='.npy')
    
    image_folder = join(dataset_path, 'OASIS_imagesTr')
    groups = {
        'OASIS_imagesTr': 414,
    }
    outdict2 = _groups_helper(image_folder, groups, prefix=None, suffix='.npy')

    output_dict = {**outdict1, **outdict2}
    
    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def sam_fed_groups():
    dataset_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/source_data/fed_dataset_lung_resampled_1mm'
    split_file_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/config_files/split_file_fed_lung_highres_1mm.json'

    train_folder = '/home/mingrui/disk_ssd1/dataset/fed_dataset_lung_resampled_1mm/imagesTr'
    test_folder = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/source_data/fed_dataset_lung/imagesTs'
    output_dict = {}
    pre_list = ['CT', 'lung', 'MICCAI']
    for pre in pre_list:
        train_list = subfiles(train_folder, prefix=pre, suffix='.npy', join=True)
        test_list = subfiles(test_folder, prefix=pre, suffix='.npy', join=True)

        output_dict['train_'+pre] = []
        for train_name in train_list:
            output_dict['train_'+pre].append({'image': train_name})
        output_dict['test_'+pre] = []
        for test_name in test_list:
            output_dict['test_'+pre].append({'image': test_name})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)


if __name__ == "__main__":
    main_random_groups()
