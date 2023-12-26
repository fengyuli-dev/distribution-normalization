## :tada: Our paper was published in NeurIPS 2023!

## About
This is the code for our paper "Test-Time Distribution Normalization For Contrastively Learned Vision-language Models". Distribution Normalization is simple enough to be easily incorporated into existing CLIP-based architectures with a few lines of code, and significantly improves the alignment of image and text representations of CLIP and its later variants. In our paper, we show that our proposed distribution normalization improves the performances in a wide range of visual-language alignment tasks, including cross-modal retrieval, zeroshot classification, and image caption evaluations. We provide examples to recreate our results below.

[![Paper](https://img.shields.io/badge/paper-2302.11084-B31B1B.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2302.11084)

## Getting Started
### Requirements
Please see <code>requirements.txt</code> for dependencies.

```
pip install -r requirements.txt
```

### Dataset
Currently, [mscoco](https://cocodataset.org/#download), [flickr8k-expert](https://www.kaggle.com/datasets/sayanf/flickr8k), [flickr8k-cf](https://www.kaggle.com/datasets/sayanf/flickr8k), [imagenet1k](https://www.image-net.org/download.php), [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html), [sun397](https://vision.princeton.edu/projects/2010/SUN/), and [stanford_cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) are supported. Please download each dataset with the corresponding link.

Modify <code>dataset_paths.py</code> to paths to your downloaded datasets. Move provided list of class names to corresponding datasets:

```
mv dataset/imagenet1k_str_labels.json [imagenet1k_path]/str_labels.json
mv dataset/cifar100_str_labels.json [cifar100_path]/str_labels.json
mv dataset/stanford_cars_str_labels.json [stanford_cars_path]/str_labels.json
```

## Replicate Zeroshot results
To run zeroshot experiments with Vanilla CLIP on cifar100, run
```
> python compute_zeroshot.py --dataset cifar100
...
cifar100 (Zero-shot Accuracy)
Top-1 Accuracy: 63.909996032714844
Top-1 STD: 3.814697265625e-06
Top-5 Accuracy: 88.70000457763672
Top-5 STD: 0.0
```

To run zeroshot experiments with CLIP + Distribution Normalization on cifar100 (using 100 random validation samples to estimate distribution mean, repeated 5 times), run
```
> python compute_zeroshot.py --dataset cifar100 --num_samples 100 --num_experiments 5 --dn
...
cifar100 (Zero-shot Accuracy)
Top-1 Accuracy: 65.03599548339844
Top-1 STD: 0.07419006526470184
Top-5 Accuracy: 89.33399963378906
Top-5 STD: 0.03878027945756912
```


## Replicate Retrieval results
To run image-to-text retrieval with Vanilla CLIP on the 1K test split of Flickr30k (using 100 random validation samples to estimate distribution mean, repeated 5 times), run
```
> python compute_retrieval.py --dataset flickr30k --num_samples 100 --num_experiments 5 --image_to_text
...
====> Doing Retrieval
Image to Text
Top-1 Accuracy: 81.30000305175781
Top-1 Std 0.0
Top-5 Accuracy: 95.0
Top-5 Std 0.0
Top-10 Accuracy: 98.5
Top-10 Std 0.0
```

To run image-to-text retrieval with CLIP + Distribution Normalization on the 1K test split of Flickr30k (using 100 random validation samples to estimate distribution mean, repeated 5 times), run
```
> python compute_retrieval.py --dataset flickr30k --num_samples 100 --num_experiments 5 --dn --image_to_text
...
====> Doing Retrieval
Image to Text
Top-1 Accuracy: 83.58000183105469
Top-1 Std 0.22271032631397247
Top-5 Accuracy: 96.14000701904297
Top-5 Std 0.04898904636502266
Top-10 Accuracy: 98.52000427246094
Top-10 Std 0.09797809273004532
```

## Replicate image captioning metric results
To run image captioning metrics experiments with Vanilla CLIP on Flickr8k-expert, run
```
> python compute_metrics.py --dataset flickr8k-expert --model regular
...
flickr8k-expert
Loaded 16992 images
266it [02:18,  1.92it/s]
CLIPScore Tau-c: 51.443
```
To run image captioning metrics experiments with CLIP + Distribution Normalization on Flickr8k-expert, run
```
> python compute_metrics.py --dataset flickr8k-expert --model dn
...
flickr8k-expert
Loaded 16992 images
266it [02:18,  1.93it/s]
CLIPScore Tau-c: 53.210
```



