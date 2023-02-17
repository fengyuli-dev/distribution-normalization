## About
This is code for our paper "Distribution Normalization: An Effortless Test-Time Augmentation for Contrastively Learned Visual-Language Models". Distribution Normalization is simple enough to be easily incorporated in existing codes with a few lines of code, and significantly improves the alignment of image and text representations of CLIP and its later variants. In our paper, we show that our proposed distribution normalization improves the performances in a wide range of visual-language alignment tasks, including cross-modal retrieval, zeroshot classification, and evaluation of image captions.

## Getting Started
### Requirements
Please see <code>requirements.txt</code> for dependencies.

<code>pip install -r requirements.txt</code>

### Dataset
Currently, [mscoco](https://cocodataset.org/#download), [flickr8k-expert](https://www.kaggle.com/datasets/sayanf/flickr8k), [flickr8k-cf](https://www.kaggle.com/datasets/sayanf/flickr8k), [imagenet1k](https://www.image-net.org/download.php), [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html), [sun397](https://vision.princeton.edu/projects/2010/SUN/), and [stanford_cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) are supported. Please download each dataset with the corresponding link.


