# Highly Shareable Multi-Task Transformer Network for Fully Automatic Pulmonary Nodule Segmentation

This repository contains a reference implementation of the algorithms described in our paper.

## Introduction

Pulmonary nodule segmentation from the lung computed tomography (CT) scan is a fundamental prerequisite for lung cancer diagnosis and therapy. However, manually screening and segmenting nodules from the entire CT scan is labor-intensive and requires a high level of expertise. Current fully automatic segmentation methods typically design and train separate detection and segmentation models, resulting in heavy models with suboptimal performance. In this study, we propose a light weight highly shareable multi-task Transformer network for fully automatic pulmonary nodule segmentation, which implements the detection and segmentation in a unified framework. Our framework makes two tasks share network parameters as many as possible for smaller model size and can be trained in an end-to-end manner for optimal performance. To alleviate the inter-task inconsistency between two different tasks, we unify the optimization objectives of detection and segmentation as the pixel-level prediction to avoid conflict in the shared feature learning. To improve the detection performance with low computational consumption, the deformable Transformer (DeTrans) is employed to build long-distance context relationships and improve the sensitivity of small early-stage pulmonary nodules. To further improve the segmentation performance, the late-cropping strategy is introduced to simultaneously concentrate on both global and local CT information for comprehensive prediction. The proposed method is validated on the aligned LUNA16 and LIDC-IDRI datasets, and experimental results demonstrate that our method outperforms the state-of-the-art pulmonary nodule detection and segmentation methods with the minimum model size.



![](./Framework.jpg)



## Prerequisite

- Python 3.8

- PyTorch 1.10.0, torchvision 0.11.1

- CUDA 11.3

- For common packages, simply run:

  ```python
  pip install -r requirements.txt
  ```

  The packages listed here may not be complete. if you run into missing packages, you may want to google and install them.

## Usage

- Dataset

  The public datasets can be downloaded from [LUNA16](https://luna16.grand-challenge.org/data/) and [LIDC_IDRI](https://www.cancerimagingarchive.net/nbia-search/?CollectionCriteria=LIDC-IDRI).

- Data loading

​	You can load the training set, validation set and test set by modifying the file `config_training.py`.

- Training	

  ```python
  python train.py --save_dir /trained_model_save_path
  ```

- Testing

  ```python
  python test.py --test_save_dir /best_trained_model_path
  ```

## Statement

This project was only designed for academic research.
