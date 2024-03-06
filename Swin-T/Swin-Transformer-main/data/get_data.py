import os
import random
import pandas as pd


def get_type(generator):
    return "nature" if generator =="nature" else "ai"


def balance_per_class(args, df):
    if args.dataset.lower() in ["FFHQ_JPEG".lower(), "SIZE".lower(), "Imagenet_JPEG".lower()]:
        return df
    df_ai = df[df["generator"] != "nature"]
    df_nature = df[df["generator"] == "nature"]
    result = pd.DataFrame(columns=df.columns)

    for i in range(1000):
        df_nature_id, df_ai_id = df_nature[df_nature["class"] == i], df_ai[df_ai["class"] == i]
        n_to_sample = min(len(df_nature_id), len(df_ai_id))
        result = pd.concat([result, df_nature_id.sample(n=n_to_sample, random_state=42), df_ai_id.sample(n=n_to_sample, random_state=42)])
    return result


def get_data(args, is_training=True, balance_val_classes=False, balance_train_classes=False):
    df = pd.read_csv(args.csv_data_path)
    if args.base_path is not None:
        df["path"] = df["path"].apply(lambda x: os.path.join(args.base_path, x))
    df["target"] = df["generator"].apply(get_type)
    if is_training:
        if args.dataset.lower() == "CLASSIC".lower():
            data_train = df[(df['mode'] == 'train') & (df["path"].apply(lambda x: x.split("/")[-5]) == args.generator)]
            data_val = df[(df['mode'] == 'val') & (df["path"].apply(lambda x: x.split("/")[-5]) == args.generator)]

        elif args.dataset.lower() == "JPEG96".lower():
            data_train_real = df[(df["mode"] == "train") & (df["generator"] == "nature") & \
                                (df["path"].apply(lambda x: x.split("/")[-5]) == args.generator) & (df["compression_rate"] == 96)]
            data_train_ai = df[(df["mode"] == "train") & (df["generator"] == args.generator)]
            data_train_ai = data_train_ai.sample(n=len(data_train_real))
            data_train = pd.concat([data_train_real, data_train_ai])

            data_val_real = df[(df["mode"] == "val") & (df["generator"] == "nature") & \
                                (df["path"].apply(lambda x: x.split("/")[-5]) == args.generator) & (df["compression_rate"] == 96)]
            data_val_ai = df[(df["mode"] == "val") & (df["generator"] == args.generator)]
            data_val_ai = data_val_ai.sample(n=len(data_val_real))
            data_val = pd.concat([data_val_real, data_val_ai])

        elif args.dataset.lower() == "SIZE_CONSTRAINED".lower():
            ### Have to be carefull with the min max sizes and the used generator
            data_train_real = df[(df["mode"] == "train") & (df["generator"] == "nature") & 
                                (df["width"] >= args.min_size) & (df["height"] >= args.min_size) & (df["width"] <= args.max_size) & (df["height"] <= args.max_size) & (df["compression_rate"] == args.jpeg_qf)]

            
            if not balance_train_classes:
                data_train_ai = df[(df["mode"] == "train") & (df["generator"] == args.generator)].sample(n=len(data_train_real), random_state=42)
            else:
                data_train_ai = df[(df["mode"] == "train") & (df["generator"] == args.generator)]
            
            data_train = pd.concat([data_train_real, data_train_ai])

            data_val_real = df[(df["mode"] == "val") & (df["generator"] == "nature") & \
                                (df["width"] >= args.min_size) & (df["height"] >= args.min_size) & (df["width"] <= args.max_size) & (df["height"] <= args.max_size) & (df["compression_rate"] == args.jpeg_qf)]
            if not balance_val_classes:
                data_val_ai = df[(df["mode"] == "val") & (df["generator"] == args.generator)].sample(n=len(data_val_real), random_state = 42)
            else:
                data_val_ai = df[(df["mode"] == "val") & (df["generator"] == args.generator)]
            data_val = pd.concat([data_val_real, data_val_ai])
        
        else:
            raise NotImplementedError("If training, args.dataset must be one of classic, jpeg96 or size_constrained")

        if balance_train_classes:
            data_train = balance_per_class(args, data_train)

        if balance_val_classes:
            data_val = balance_per_class(args, data_val)

        print(f"{len(data_train)} training data")
        print(f"{len(data_val)} validation data!")
         
        return data_train, data_val

    else:
    # Data for evaluating the model
        if args.dataset.lower() == "FFHQ_JPEG".lower():
            # give all ffhq paths back, the ffhq are then respectively compressed with the args.compression argument (to do in get_transform)
            data_val = df

        elif args.dataset.lower() == "SIZE".lower():
            # in this experiment, we search how the classifier classifies real data that it didnt see in training, for a given size interval
            # Taking data that are nature, and where the generator directory is different than the generator directory the model was trained on, or the mode is validation
            data_val = df[(df["generator"] == "nature") & ((df["mode"] == "val") | (df["path"].apply(lambda x: x.split("/")[-5]) != args.generator_trained_on))]
            data_val = data_val[(data_val["width"] >= args.min_width) & (data_val["height"] >= args.min_height) & (data_val["width"] <= args.max_width) & (data_val["height"] <= args.max_height)]
            if args.min_qf is not None:
                data_val = data_val[data_val["compression_rate"] >= args.min_qf]

        elif args.dataset.lower() == "IMAGENET_JPEG".lower():
            # in this experiment, we take all ai generated data
            data_val = df[(df["path"].apply(lambda x: x.split("/")[-5]) == args.generator) & (df["mode"] == "val") & (df["compression_rate"] >= args.jpeg_qf)]

        elif args.dataset.lower() == "CLASSIC".lower():
            # classical cross generator validation,evaluates the model trained on validation data of args.generator
            data_val = df[(df["path"].apply(lambda x: x.split("/")[-5]) == args.generator) & (df["mode"] == "val")]
        
        else:
            raise NotImplementedError("If not training (testing/validating), args.dataset must be one of ffhq_jpeg, imagenet_jpeg, size, classic. See ReadMe markdown file ")

        if balance_val_classes:
            data_val = balance_per_class(args, data_val)
        
        print(f"{len(data_val)} val data!")
        return data_val