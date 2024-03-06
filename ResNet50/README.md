# ResNet50 

Fork from [timm](https://pypi.org/project/timm/0.6.12/)
```bash
cd pytorch-image-models-0.6.12
```

## Training

To start a training on the raw GenImage dataset: 
```bash
sh ./distributed_train.sh 2 classic <path_to_genimage_csv> <train_subset> \
--class-map <../../class_map.txt> -b 128 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel \
--reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --jsd --dist-bn reduce --num-classes 2 \
--output <result_folder> --experiment <exp_name> \
--base_path <prepended_to_paths_in_csv> \
--resume <checkpoint.pth.tar>
```

To start a training only using JPEG(qf=96) images: 
```bash
sh ./distributed_train.sh 2 jpeg96 <path_to_genimage_csv> <train_subset> \
--class-map <../../class_map.txt> -b 128 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel \
--reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --jsd --dist-bn reduce --num-classes 2 \
--output <result_folder> --experiment <exp_name> \
--base_path <prepended_to_paths_in_csv> --jpeg_qf 96 \
--resume <checkpoint.pth.tar>
```

To start a training only using JPEG(qf=96) images and including size constrain: 
```bash
sh ./distributed_train.sh 2 size_constrained <path_to_genimage_csv> <train_subset> \
--class-map <../../class_map.txt> -b 128 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel \
--reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --jsd --dist-bn reduce --num-classes 2 \
--output <result_folder> --experiment <exp_name> \
--base_path <prepended_to_paths_in_csv> \
--balance_train_classes --min_size <lower_bound> --max_size <upper_bound> --cropsize <lower_bound> --jpeg_qf 96 \
--resume <checkpoint.pth.tar>
```

NOTE: <br>
--resume is optional <br>
--base_path is the path to the GenImage download and is prepended to the paths in the CSV file

## Validation

example of Cross-Generator-Validation:

```bash
python validate.py \
classic <path_to_genimage_csv> --generator <eval_subset> \
--base_path <prepended_to_paths_in_csv> --class-map <../../class_map.txt> \
--results-file <results.csv> \
--model resnet50 --checkpoint <eval_checkpoint.pth.tar> --num-classes 2 --amp \
--jpeg_qf <qf> --resize <size> --cropsize <size>
```

NOTE: <br>
--jpeg_qf, --resize, --cropsize are optional <br>
--compress_natural if you want to compress natural images as well <br>
(see args for more possible experiments)









