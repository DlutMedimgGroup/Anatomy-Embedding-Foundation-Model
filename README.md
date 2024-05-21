# Anatomy Embedding Foundation Model



## Installation

- Create a virtual environment `conda create -n monai` and activate it `conda activate monai`
- Install PyTroch `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- `git clone https://github.com/DlutMedimgGroup/Anatomy-Embedding-Foundation-Model`
- Enter the project folder `cd Anatomy-Embedding-Foundation-Model` and run `pip install -e .`


## Get Started

Download the [model checkpoint](https://1drv.ms/f/s!AiAogjEIFaXOgulqdHCFV3p4O24fQg?e=qWkmg5) and place it at e.g. `./trained_models/abdomen_foundation.pth` and `./trained_models/downstream_seg.pth`

1. Test the segmentation model on the sample images

    ``` shell
    python run_inference.py -c ./config_files/downseg_20.toml
    ```

    The results will be stored at `./data/inference/downstream`

2. Test the segmentation model on your images

    You need to change the config file to set the test dataset. The method is to change the `split_filename` in the Inference section of the configuration file `downseg_20.toml`. If you want to create a new `split file`, you can use `./tools/split_file_creator.py`

## Downsteam Segmentation Network Training

Preprocess the training data with `./tools/preprocess.py`

Create a new `split file` with `./tools/split_file_creator.py`

Firstly, set the `split_filename` in the Training section of the configuration file `downseg_20.toml`.

Run training with:
``` shell
python run_training.py -c ./config_files/downseg_20.toml
```

