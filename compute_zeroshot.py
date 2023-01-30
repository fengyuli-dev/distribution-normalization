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
import os
import scipy.stats
import clipscore
import sys
from clip_zeroshot import zeroshot_prediction
from clip_zeroshot import IMAGENET1K_DIR, IMAGE_CAPTION_METRICS, CIFAR100_DIR, SUN397_DIR, STANFORDCARS_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn('Running on CPU.')


def main(args):
    print(f'{args.dataset} (Zero-shot Accuracy)')
    model_name = args.model
    if model_name == 'regular' or model_name == 'first':
        model = clipscore.OriginalCLIPScore()
    elif model_name == 'batch':
        model = clipscore.BatchCLIP()
    model.to(device)

    images = []
    labels = []
    captions = []
    if args.dataset == 'cifar100':
        with open(CIFAR100_DIR + '/str_labels.json', 'r') as f:
            str_labels = json.load(f)
        captions = [f'a photo of a {s}.' for s in str_labels]
        validation_dir = os.path.join(CIFAR100_DIR, 'validation')
        for label in range(100):
            image_folder = validation_dir + '/' + str(label)
            current_images = os.listdir(image_folder)
            for image in current_images:
                images.append(image_folder + '/' + image)
                labels.append(label)

    if args.dataset == 'imagenet1k':
        with open(IMAGENET1K_DIR + '/str_labels.json', 'r') as f:
            str_labels = json.load(f)
        captions = [f'a photo of a {s}.' for s in str_labels]
        training_dir = os.path.join(IMAGENET1K_DIR, 'validation')
        for label in range(1000):
            image_folder = training_dir + '/' + str(label)
            current_images = os.listdir(image_folder)
            for image in current_images:
                images.append(image_folder + '/' + image)
                labels.append(label)

    if args.dataset == 'sun397':
        class_names = []
        with open(SUN397_DIR + '/ClassName.txt', 'r') as fb:
            for line in fb:
                class_names.append(line.replace(
                    '\n', '').strip())
        captions = [
            f'a photo of a {s.replace("_", " ").split("/")[2]}.' for s in class_names]
        # print(captions)
        for i, class_name in enumerate(class_names):
            image_folder = SUN397_DIR + class_name
            current_images = os.listdir(image_folder)[-50:]
            for image in current_images:
                images.append(image_folder + '/' + image)
                labels.append(i)

    if args.dataset == 'stanford_cars':
        with open(STANFORDCARS_DIR + '/str_labels.json', 'r') as f:
            classes = json.load(f)
        captions = [f'a photo of a {s}' for s in classes]
        with open(STANFORDCARS_DIR + '/devkit/train_perfect_preds.txt', 'r') as f:
            for line in f:
                labels.append(int(line.replace('\n', '').strip())-1)
        image_folder = STANFORDCARS_DIR + '/cars_train'
        current_images = sorted(os.listdir(image_folder))
        for image in current_images:
            images.append(image_folder + '/' + image)

        # load the model
    if model_name == 'batch':
        model.load_state_dict(torch.load('BatchCLIP.pt')['model_state_dict'])

    acc = zeroshot_prediction(model, images, captions, labels, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="cifar100",
                        choices=["cifar100", "imagenet1k", "sun397", "stanford_cars"], type=str)
    parser.add_argument('--model', default='first',
                        choices=['batch', 'regular', 'first'], type=str)
    parser.add_argument('--universal_mean', action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False, help="normalize then take mean")
    parser.add_argument('-s', '--scalar', type=float, default=0.25)
    args = parser.parse_args()
    main(args)
