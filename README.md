Physics-Coupled Frequency Dynamic Adaptation Network for Domain Generalized Underwater Object Detection

This repository contains the official PyTorch implementation for our paper accepted by **ACM Multimedia (ACM MM) 2025**:
 - [Physics-Coupled Frequency Dynamic Adaptation Network for Domain Generalized Underwater Object Detection](https://dl.acm.org/doi/10.1145/3746027.3755829)

# 📂 Installation

 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/White-cat-ed/HyUOD
   cd HyUOD
   ```
 - Set up the environment using Conda:
   ```Shell
   # Create a conda environment with Python 3.8
   conda create -n hyuod python=3.8 -y
   conda activate hyuod
   ```
 - Install PyTorch (2.1.1+cu118) and TorchVision:
   ```Shell
   pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```
 - Install the remaining packages from `requirements.txt`:
   ```Shell
   pip install -r requirements.txt
   ```

# 📦 Data Preparation

First, download your target underwater object detection dataset and organize it into the standard YOLO dataset format. The initial directory structure should look like this:

```text
dataset_root/
├── images/
│   ├── train/
│   └── test/
└── labels/
    ├── train/
    └── test/
```

To accelerate the training process, we pre-generate the transmission map (`t`) and global atmospheric light (`A`). Run the `ta_generate.py` script to generate these physical parameters:

```Shell
python ta_generate.py /path/to/dataset_root/images /path/to/dataset_root
```

After generation, your dataset directory structure should be updated to:

```text
dataset_root/
├── images/
│   ├── train/
│   └── test/
├── labels/
│   ├── train/
│   └── test/
├── t/
│   ├── train/
│   └── test/
└── a/
    ├── train/
    └── test/
```

Finally, create your dataset YAML file. Here is an example using the **DUO dataset**:

```yaml
path: /path/to/dataset_root  # dataset root dir
train: images/train 
train_t: t/train 
train_a: a/train 
val: images/test 
nc: 4
names:
  0: holothurian
  1: echinus
  2: scallop
  3: starfish
```

# 🚀 Usage

We provide a unified entry point `main.py` for all operations. Alternatively, you can run individual scripts.

## 1. Data Preparation
To accelerate the training process, we pre-generate the transmission map (`t`) and global atmospheric light (`A`). Run the following command:

```Shell
python main.py prep /path/to/dataset_root/images /path/to/dataset_root
```
*Or use the original script:* `python ta_generate.py /path/to/dataset_root/images /path/to/dataset_root`

After generation, your dataset directory structure should be:
```text
dataset_root/
├── images/ (train, test)
├── labels/ (train, test)
├── t/      (train, test)
└── a/      (train, test)
```

## 2. Training
Train the model using the unified entry point:
```Shell
python main.py train train_yaml/hyuod.yaml /path/to/your/data_yaml.yaml --epochs 400 --batch 32 --device 0
```
*Or use the original script:* `python train.py train_yaml/hyuod.yaml /path/to/your/data_yaml.yaml`

## 3. Evaluation
Evaluate the trained model:
```Shell
python main.py val weights/DUO.pt /path/to/your/data_yaml.yaml --imgsz 640 --device 0
```
*Or use the original script:* `python val.py weights/DUO.pt /path/to/your/data_yaml.yaml`

# 🛠 Key Improvements
- **Unified CLI**: Added `main.py` for a more streamlined workflow.
- **Flexible Training**: `train.py` and `val.py` now support more command-line arguments (epochs, batch size, device, optimizer, etc.).
- **Robust Preprocessing**: `ta_generate.py` now includes better path handling and error warnings.
- **Improved Logging**: Better console output during training and validation.

# 📜 Citation

If you use HyUOD or this code base in your work, please cite our paper:

```bibtex
@inproceedings{10.1145/3746027.3755829,
author = {Luo, Linxuan and Mu, Pan and Bai, Cong},
title = {Physics-Coupled Frequency Dynamic Adaptation Network for Domain Generalized Underwater Object Detection},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {[https://doi.org/10.1145/3746027.3755829](https://doi.org/10.1145/3746027.3755829)},
doi = {10.1145/3746027.3755829},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {2284–2293},
numpages = {10},
keywords = {deep learning, domain generalized underwater object detection, hypernetwork, underwater object detection},
location = {Dublin, Ireland},
series = {MM '25}
}
```
```
