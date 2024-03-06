# 🌟 Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets

This 🖥️📦 [Repository](https://github.com/gendetection/UnbiasedGenImage) corresponds to our 📚📄 [Paper](https://www.unbiased-genimage.org) towards Biases in datasets for AI-Generated Images Detection. As discussed detailed in the paper, experiments are examined on the [GenImage](https://genimage-dataset.github.io/) dataset. 

### Download

⬇️ We provide an easy GenImage download here: [TODO](https://www.unbiased-genimage.org). Furthermore, we removed corrupted files in the GenImage download and added a metadata CSV. This CSV is needed for our training and validation code and contains additional information like content classes of each image which is not part of the original dataset.

## Code details

We provide [Code](https://github.com/gendetection/UnbiasedGenImage) for training and validating ResNet50 and Swin-T detectors. This aims to show that:

1. Detectors trained on the raw GenImage dataset actually learn from existing Biases in compression and image size.
2. Mitigating these Biases leads to significantly improved Cross-Generator Performance and Robustness towards JPEG-Compression, achieving state-of-the-art results.

Same as in the original GenImage paper, we use forks from [timm](https://pypi.org/project/timm/0.6.12/) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We just changed the dataset (create_dataset.py) to be more suitable for our experiments. This dataset uses get_data.py for selecting the right data from the csv file and get_transform.py for transformations like JPEG-compression that are applied before the original transformations/augmentations. More details for how to start experiments can be found in the corresponding detector folders.

<br>

To do inference on own datasets, you have to create a CSV file and slightly adjust get_data.py as we did for the ffhq dataset.


## Results

### ResNet50

<p align="center">
  <img src="results/results_resnet.png" width="80%" />
  <br>
  <em>Cross-Generator Performance when training ResNet50 on constrained dataset</em>
</p>

<br>

<p align="center">
  <img src="results/results_resnet_diff.png" width="80%" />
  <br>
  <em>Difference to when training on raw dataset</em>
</p>

<br><br>

### Swin-T

<p align="center">
  <img src="results/results_swin.png" width="80%" />
  <br>
  <em>Cross-Generator Performance when training Swin-T on constrained dataset</em>
</p>

<br>

<p align="center">
  <img src="results/results_swin_diff.png" width="80%" />
  <br>
  <em>Difference to when training on raw dataset</em>
</p>
