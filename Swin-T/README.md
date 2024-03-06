# Swin-T 

Fork from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

### Setup:
```bash
conda env create -f swin_environment.yml -n <name>
conda activate <name>
cd Swin-Transformer-main
```

## Training

To start a training on the raw GenImage dataset: 

```bash
python3 -m torch.distributed.launch --nproc_per_node=2 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--csv_data_path <path_to_genimage_csv> --generator <train_subset> --dataset classic \
--base_path <prepended_to_paths_in_csv> --output <result_folder> \
--resume <optional_checkpoint.pth>
```

To start a training only using JPEG(qf=96) images and including size constrain:

```bash
python3 -m torch.distributed.launch --nproc_per_node=2 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--csv_data_path <path_to_genimage_csv> --generator <train_subset> --dataset size_constrained \
--base_path <prepended_to_paths_in_csv> --output <result_folder> --batch-size 128 \
--balance_train_classes --min_size <lower_bound> --max_size <upper_bound> --cropsize <lower_bound> --jpeg_qf 96 \
--resume <checkpoint.pth>
```

NOTE: <br>
--resume is optional <br>
--base_path is the path to the GenImage download and is prepended to the paths in the CSV file

## Validation

example of Cross-Generator-Validation:

```bash
python3 -m torch.distributed.launch --nproc_per_node=1 validate.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 128 \
--csv_data_path <path_to_genimage_csv> --dataset classic --generator <eval_subset> \
--base_path <prepended_to_paths_in_csv> --pretrained <eval_checkpoint.pth> \
--output <result_dir> --tag <exp_name> \
--jpeg_qf <qf> --resize <size> --cropsize <size>
```

NOTE: <br>
--jpeg_qf, --resize, --cropsize are optional <br>
--compress_natural if you want to compress natural images as well <br>
(see args for more possible experiments)

