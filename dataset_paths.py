'''
Put the paths to your datasets here.
'''
from pathlib import Path

IMAGE_CAPTION_METRICS = '/share/cuvl/image_caption_metrics'
CIFAR100_DIR = '/share/cuvl/image_caption_metrics/cifar-100-python'
IMAGENET1K_DIR = Path('/share/cuvl/image_caption_metrics/imagenet-1k')
SUN397_DIR = Path('/share/cuvl/image_caption_metrics/SUN397')
STANFORDCARS_DIR = Path('/share/cuvl/image_caption_metrics/stanford_cars')
IMAGE_CAPTION_METRICS = Path('/share/cuvl/image_caption_metrics')
FLICKR8K_DIR = Path(IMAGE_CAPTION_METRICS, 'flickr8k')
FLICKR30K_DIR = Path(IMAGE_CAPTION_METRICS, 'flickr30k')
FLICKR30K_SENTENCES = Path(IMAGE_CAPTION_METRICS, 'flickr30k_sentences')
MSCOCO_DIR = Path(IMAGE_CAPTION_METRICS, 'MSCOCO_VAL2014')
PASCAL_DIR = Path(IMAGE_CAPTION_METRICS, 'pascal')
