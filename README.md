# Data Centric Mindset in Bioimaging AI

This repository provides a general, convenient, and suitable **Data-Centric framework** for processing large amounts of unlabeled **bioimaging data**. It consists of four stages:

1. **Pre-training:** Perform self-supervised pre-training on the entire raw dataset, using the trained encoder to build task-specific models.

2. **Understanding the dataset:** Select the most representative images for annotation, rather than choosing randomly.

3. **Iteratively hunting for mistakes and hard cases:** Gradually improve the model by selecting samples with high uncertainty and adding them to the training set.

4. **Monitoring performance and fine-tuning:** Evaluate the model's performance on unlabeled images and perform additional annotations to fine-tune the model when necessary.


# Usage
Using vascular imaging data in (Z, Y, X) dimensions as an example, this framework can be extended to other dimensional bioimaging data. The directory structure is as follows:
```
Bio_Data_Centric
├── data
│   ├──core
│   │    ├── image
│   │    └── label
│   ├──critical
│   │    ├── image
│   │    └── label
│   ├──origin
│   ├──rca
│   │    ├── image
│   │    └── label
│   ├──train
│   └── process.py
├── mae
├── RCA
└── TransUNet
```
You can run `./data/process.py` to split the original 3D images in `./data/origin` into slices and store them in `./data/train`.
## 1. Pre-training

Run the pretrain script. We use MAE([Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)) as the pre-training model, and the pre-training instructions are in `./mae/PRETRAIN.md`. Since the number of data channels is 1, set `in_chans` in `./mae/models_mae.py` to 1.
```
sh ./mae/pretrain.sh
```

## 2. Understanding the dataset

Run the core set select script. Label core set and store them in `./data/core/label`
```
sh ./mae/core_select.sh
```
## 3. Iteratively hunting for mistakes and hard cases

1. Run the train script on core set and save the weight. We chose [TransUNet](https://arxiv.org/pdf/2102.04306) as segmentation model.
	```
	sh ./TransUNet/train.sh
	```
2. Then, run the critical set select script. Select valuable samples from the critical set, annotate them, and place them in `./data/critical`.
	```
	sh ./TransUNet/critical_select.sh
	```
3. Run the train script again on the core set and critical set, making sure to set `add_path=./data/critical`. You can repeat step 2 and 3.
	```
	sh ./TransUNet/train.sh
	```
4. Run the inference script.
	```
	sh ./TransUNet/inference.sh
	```
## 4. Monitoring performance and fine-tuning:
1. Run the RCA_train script with prediction from inference. We employed RCA ([Reverse Classification Accuracy](https://arxiv.org/pdf/1702.03407)) to predict the accuracy of image segmentation.
	```
	python ./RCA/RCA_train.py
	```
2. Run the RCA_eval script to evaluate the accuracy of the model's inference. For samples with lower accuracy, they can be annotated and used for model fine-tuning.
	```
	python ./RCA/RCA_eval.py
	```

## Reference

1. [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306)
2. [Reverse Classification Accuracy](https://arxiv.org/pdf/1702.03407)
3. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)