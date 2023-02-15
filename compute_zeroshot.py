'''
Computes metrics for Flickr8K.
'''
from pathlib import Path
import pandas as pd
import argparse
import warnings
import torch
import numpy as np
import json
import scipy
import os
import clipscore
import torch.nn.functional as F
from dataset_paths import IMAGENET1K_DIR, IMAGE_CAPTION_METRICS, CIFAR100_DIR, SUN397_DIR, STANFORDCARS_DIR
from torchmetrics import Accuracy
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn('Running on CPU.')
# the coefficient for distribution normalization
LAMBDA = 0.25


def zeroshot_prediction(model, images, val_images, captions, labels, device, args):
    model.eval().to(device)
    clip = model.clip
    clip.to(device)
    image_embeddings = clipscore.extract_all_images(
        images, clip, device, normalize=False).cpu()
    caption_embeddings = clipscore.extract_all_captions(
        captions, clip, device, normalize=False).cpu()
    top1s = []
    top5s = []
    # repeat experiments
    for _ in range(args.num_experiments):
        _image_embeddings = image_embeddings
        _caption_embeddings = caption_embeddings
        if args.dn:
            # randomly sample args.num_samples images from the validation set
            # to estimate the mean
            val_embeddings = clipscore.extract_all_images(
                random.sample(val_images, args.num_samples), clip, device, normalize=False).cpu()
            # perforrm distribution normalization
            _image_embeddings = image_embeddings - \
                torch.mean(val_embeddings, dim=0).cpu() * LAMBDA
            _caption_embeddings = caption_embeddings - \
                torch.mean(caption_embeddings, dim=0).cpu() * LAMBDA
        _image_embeddings = F.normalize(_image_embeddings)
        _caption_embeddings = F.normalize(_caption_embeddings)
        sims = []
        for i in range(0, _image_embeddings.size(0), 32):
            sims.append(_image_embeddings[i:i+32] @ _caption_embeddings.T)

        sims = torch.cat(sims, dim=0)
        top1 = Accuracy(top_k=1)(sims, torch.FloatTensor(labels).long())
        top5 = Accuracy(top_k=5)(sims, torch.FloatTensor(labels).long())
        top1s.append(top1*100)
        top5s.append(top5*100)
    print(f'Top-1 Accuracy: {np.mean(top1s)}')
    print(f'Top-1 STD: {np.std(top1s)}')
    print(f'Top-5 Accuracy: {np.mean(top5s)}')
    print(f'Top-5 STD: {np.std(top5s)}')
    return top1


def main(args):
    print(f'{args.dataset} (Zero-shot Accuracy)')
    model = clipscore.OriginalCLIPScore()
    model.to(device)

    images = []
    labels = []
    captions = []
    val_images = []

    # load cifar100 dataset
    if args.dataset == 'cifar100':
        with open(Path(CIFAR100_DIR, 'str_labels.json'), 'r') as f:
            str_labels = json.load(f)
        captions = [f'a photo of a {s}.' for s in str_labels]
        test_dir = Path(CIFAR100_DIR, 'validation')
        for label in range(100):
            image_folder = Path(test_dir, str(label))
            current_images = os.listdir(image_folder)
            for image in current_images:
                images.append(Path(image_folder, image).as_posix())
                labels.append(label)
        validation_dir = Path(CIFAR100_DIR, 'training')
        for label in range(100):
            image_folder = Path(validation_dir, str(label))
            current_images = os.listdir(image_folder)
            for image in current_images:
                val_images.append(Path(image_folder, image).as_posix())

    # load imagenet1k dataset
    if args.dataset == 'imagenet1k':
        with open(Path(IMAGENET1K_DIR, 'str_labels.json'), 'r') as f:
            str_labels = json.load(f)
        captions = [f'a photo of a {s}.' for s in str_labels]
        test_dir = Path(IMAGENET1K_DIR, 'validation')
        for label in range(1000):
            image_folder = Path(test_dir, str(label))
            current_images = os.listdir(image_folder)
            for image in current_images:
                images.append(Path(image_folder, image).as_posix())
                labels.append(label)

        validation_dir = Path(IMAGENET1K_DIR, 'training')
        for label in range(1000):
            image_folder = Path(validation_dir, str(label))
            current_images = os.listdir(image_folder)
            for image in current_images:
                val_images.append(Path(image_folder, image).as_posix())

    # load sun397 dataset
    if args.dataset == 'sun397':
        class_names = []
        with open(Path(SUN397_DIR, 'ClassName.txt'), 'r') as fb:
            for line in fb:
                class_names.append(line.replace(
                    '\n', '').strip())
        captions = [
            f'a photo of a {s.replace("_", " ").split("/")[2]}.' for s in class_names]
        # print(captions)
        for i, class_name in enumerate(class_names):
            image_folder = Path(SUN397_DIR, class_name[1:])
            test_images = os.listdir(image_folder)[-50:]
            validation_images = os.listdir(image_folder)[:50]
            for image in test_images:
                images.append(Path(image_folder, image).as_posix())
                labels.append(i)
            for image in validation_images:
                val_images.append(Path(image_folder, image).as_posix())

    # load stanford_cars dataset
    if args.dataset == 'stanford_cars':
        with open(Path(STANFORDCARS_DIR, 'str_labels.json'), 'r') as f:
            classes = json.load(f)
        captions = [f'a photo of a {s}' for s in classes]
        image_folder = Path(STANFORDCARS_DIR, 'cars_test')
        test_data = scipy.io.loadmat(
            Path(STANFORDCARS_DIR, 'cars_test_annos_withlabels.mat'))['annotations'][0]
        for d in test_data:
            images.append(Path(image_folder, d[5][0]).as_posix())
            labels.append(d[4][0][0] - 1)
        val_data = scipy.io.loadmat(
            Path(STANFORDCARS_DIR, 'devkit/cars_train_annos.mat'))['annotations'][0]
        image_folder = Path(STANFORDCARS_DIR, 'cars_train')
        for d in val_data:
            val_images.append(Path(image_folder, d[5][0]).as_posix())

    zeroshot_prediction(
        model, images, val_images, captions, labels, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="cifar100",
                        choices=["cifar100", "imagenet1k", "sun397", "stanford_cars"], type=str)
    parser.add_argument('--dn', action="store_true", default=False)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--num_experiments', default=5, type=int)
    args = parser.parse_args()
    main(args)
