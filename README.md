## Project 

This repository presents my final solution for a 2022 Kaggle competition (https://www.kaggle.com/c/mva-recvis-2022) on fine-grained image classification. This competition was in the MVA Object Recognition and Computer Vision class. The goal of this competition was to design a model that has maximum test accuracy on Image Classiﬁcation of a subset of the Birds200 dataset. I achieved 90.322% of accuracy in the public leaderboard (8th place /131), and 91.160% of accuracy in the private leaderboard (9th place /131).

## Task Description - Object recognition and computer vision 2022/2023 (MVA)

### Assignment 3: Image classification 

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model
Run the script `main.py` to train your model.

Modify `main.py`, `model.py` and `data.py` for your assignment, with an aim to make the validation score better.

- By default the images are loaded and resized to 64x64 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Adaptation done by Gul Varol: https://github.com/gulvarol
