# 🌟 Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets

This 🖥️📦 [Repository](https://github.com/gendetection/UnbiasedGenImage) corresponds to our 📚📄 [Paper](https://arxiv.org/abs/2403.17608) towards Biases in datasets for AI-Generated Images Detection. As discussed detailed in the paper, experiments are examined on the [GenImage](https://genimage-dataset.github.io/) dataset. 

# Unbiased GenImage dataset

### 1) Download

To use our Unbiased GenImage dataset, you first need to download the original GenImage dataset and our additional metadata CSV which contains additional information about jpeg QF, size and content of each image. This CSV is needed for our training and validation code.

⬇️ We provide an easy GenImage (and metadata CSV) download here (~500GB): [DOWNLOAD](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FAKDIHF). Furthermore, we removed corrupted files from the BAIDU GenImage download.
<br>
Use our download-script like this, since the web interface doesn't allow downloading all files together:

```bash
python download_genimage.py <--continue> <--destination {path}>
```

- `--continue`: Optional. Skip files if they already exist. Default is to start a new download.
- `--destination {path}`: Optional. Specify a custom directory where the files should be downloaded. Default is ./GenImage_download
<br>
Then get the final zip file using:

```bash
cat GenImage.z* > ../GenImage_restored.zip
```

ℹ️ NOTE: By now, there's an easy GenImage download on [Google Drive](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS). We recommend downloading the GenImage dataset there and only downloading the metadata.csv from our dataverse. ℹ️

### 2) Remove biases:

As shown in our training code of the detectors (-> get_data.py and get_transform.py), you can create our Unbiased Genimage dataset by selecting the subset of images in a specific size range (or by content classes). Then align the jpeg QG using jpeg_augment.py.

Example to create the (by size and compression) unbiased Wukong (512x512 px) subset:
```bash
df = pd.read_csv("metadata.csv")
df_unbiased_natural = df[ (df["generator"] == "nature") & (df["width"] >= 450) & (df["height"] >= 450) & (df["width"] <= 550) & (df["height"] <= 550) & (df["compression_rate"] == 96)]
df_unbiased_ai = df[ (df["generator"] == "wukong") ]
df_unbiased = pd.concat([df_unbiased_natural, df_unbiased_ai])
```



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
