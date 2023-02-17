## About
This is the code for our paper "Distribution Normalization: An Effortless Test-Time Augmentation for Contrastively Learned Visual-Language Models". Distribution Normalization is simple enough to be easily incorporated into existing CLIP-based architectures with a few lines of code, and significantly improves the alignment of image and text representations of CLIP and its later variants. In our paper, we show that our proposed distribution normalization improves the performances in a wide range of visual-language alignment tasks, including cross-modal retrieval, zeroshot classification, and image caption evaluations. We provide examples to recreate our results below.

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
To run retrieval experiments with Vanilla CLIP on the 1K test split of Flickr30k, run
```
> python compute_metrics.py --dataset flickr30k --model regular --retrieval
...
====> Doing Retrieval

```

To run retrieval experiments with CLIP + Distribution Normalization on the 1K test split of Flickr30k, run
```
> python compute_metrics.py --dataset flickr30k --model first --retrieval
...
====> Doing Retrieval

```



