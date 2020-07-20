## EdgeConnect with different mask type strategies
### Introduction:
The solution is based on https://github.com/knazeri/edge-connect/tree/master/src, which is mainly focus on images with small size. 

* Our solution categorize images into three different groups according to their masks. 
	* For box mask, the image will be resize to three smaller size in order to have the more structure information and be able to pass our model. After resizing back to original size, we compose three image with different weight into one image. 
	* For large images with other masks, we cut image into many patches and let them pass model one by one. After all, we put them back to original position. This method might cause some color difference. So the patch we used will be as large as the server GPU can accept.
	* For small images with other masks, we directly pass the model.

## Prerequisites
- Python 3.6
- PyTorch 1.3.1
- NVIDIA GPU + CUDA cuDNN

## Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets

### 1) Images
The author use  [Paris Street-View](https://github.com/pathak22/context-encoder) datasets to train the model, which help our model to perform better since it could inpaint with more detail. So we continue training on ADE20K dataset based on the .pth file author provide. 

* First, run[`scripts/split.py`](scripts/flist.py) to separate the img and mask in the data set into different designated folders. 

```bash
mkdir datasets
cd datasets
mkdir test_img
mkdir test_mask
cd ../
python ./scripts/split.py --data_dir path_to_ADE20K_test_set
```

* Then, run [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set file lists. 

	* For the **training** set, we only need to generate image flist. 

	```bash
	python ./scripts/flist.py --path path_to_training_img --output ./datasets/train_img.flist
	```

	* For **validation** and **test** set, we need to generate image flist and mask flist. For example, to generate the test set file list on ADE20K dataset run:

	```bash
	python ./scripts/flist.py --path path_to_test_img --output ./datasets/test_img.flist
	python ./scripts/flist.py --path path_to_test_mask --output ./datasets/test_mask.flist
	```

### Quick Test

We already set all the parameters and test data as default. If you only want to **test the AIM2020 Inpainting Track 1**. Run:

```bash
python test.py
```

The test result will be saved in ./result.

## Getting Started

### 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/knazeri/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

EdgeConnect is trained in three stages: 1) training the edge model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example, to train model 1. We should execute following command
```bash
python train.py --model 1 --checkpoints ./checkpoints
```

Convergence of the model differs from dataset to dataset. For example Places2 dataset converges in one of two epochs, while smaller datasets like CelebA require almost 40 epochs to converge. You can set the number of training iterations by changing `MAX_ITERS` value in the configuration file.

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py \
  --model [stage] \(Normally, it shou)
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

For example, Please download our .pth, put into ./checkpoints/ and run: 
```bash
python test.py --model 3 --checkpoints ./checkpoints --input ./datasets/test_img.flist --mask ./datasets/test_mask.flist --output ./result
```
By default `test.py` script is run on stage 3 (`--model=3`).

### Model Configuration

The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:

#### General Model Configurations

Option          | Description
----------------| -----------
MODE            | 1: train, 2: test, 3: eval
MODEL           | 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
MASK            | 1: random block, 2: half, 3: external, 4: external + random block, 5: external + random block + half
EDGE            | 1: canny, 2: external
NMS             | 0: no non-max-suppression, 1: non-max-suppression on the external edges
SEED            | random number generator seed
GPU             | list of gpu ids, comma separated list e.g. [0,1]
DEBUG           | 0: no debug, 1: debugging mode
VERBOSE         | 0: no verbose, 1: output detailed statistics in the output console

#### Loading Train, Test and Validation Sets Configurations

Option          | Description
----------------| -----------
TRAIN_FLIST     | text file containing training set files list
VAL_FLIST       | text file containing validation set files list
TEST_FLIST      | text file containing test set files list
TRAIN_EDGE_FLIST| text file containing training set external edges files list (only with EDGE=2)
VAL_EDGE_FLIST  | text file containing validation set external edges files list (only with EDGE=2)
TEST_EDGE_FLIST | text file containing test set external edges files list (only with EDGE=2)
TRAIN_MASK_FLIST| text file containing training set masks files list (only with MASK=3, 4, 5)
VAL_MASK_FLIST  | text file containing validation set masks files list (only with MASK=3, 4, 5)
TEST_MASK_FLIST | text file containing test set masks files list (only with MASK=3, 4, 5)

#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.0   | adam optimizer beta1
BETA2                  | 0.9   | adam optimizer beta2
BATCH_SIZE             | 2    | input batch size 
INPUT_SIZE             | 512 | input image size for training. (0 for original size)
SIGMA                  | 2     | standard deviation of the Gaussian filter used in Canny edge detector </br>(0: random, -1: no edge)
MAX_ITERS              | 2e6   | maximum number of iterations to train the model
EDGE_THRESHOLD         | 0.5   | edge detection threshold (0-1)
L1_LOSS_WEIGHT         | 1     | l1 loss weight
FM_LOSS_WEIGHT         | 10    | feature-matching loss weight
STYLE_LOSS_WEIGHT      | 1     | style loss weight
CONTENT_LOSS_WEIGHT    | 1     | perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT| 0.01  | adversarial loss weight
GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN
GAN_POOL_SIZE          | 0     | fake images pool size
SAVE_INTERVAL          | 1000  | how many iterations to wait before saving model (0: never)
EVAL_INTERVAL          | 0     | how many iterations to wait before evaluating the model (0: never)
LOG_INTERVAL           | 10    | how many iterations to wait before logging training loss (0: never)
SAMPLE_INTERVAL        | 1000  | how many iterations to wait before saving sample (0: never)
SAMPLE_SIZE            | 12    | number of images to sample on each samling interval



## Citation

If you find this repository useful, please cite:

```bash
@misc{AIM2020RealSR,
  author = {Haopeng Ni},
  title = {AIM2020-Inpainting},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/juechen-zzz/AIM2020-Inpainting}},
}
```

