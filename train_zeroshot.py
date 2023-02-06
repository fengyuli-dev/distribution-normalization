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
from clip_zeroshot import zeroshot_train_fast
sys.path.append('../')


IMAGE_CAPTION_METRICS = Path('/share/cuvl/image_caption_metrics')
CIFAR100_DIR = '/share/cuvl/image_caption_metrics/cifar-100-python'

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn('Running on CPU.')


def main(args):
    print(f'{args.dataset} (Zero-shot Accuracy)')
    model_name = args.model
    if model_name == 'regular':
        model = clipscore.OriginalCLIPSore()
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
        training_dir = os.path.join(CIFAR100_DIR, 'training')
        for label in range(100):
            image_folder = training_dir + '/' + str(label)
            current_images = os.listdir(image_folder)
            for image in current_images:
                images.append(image_folder + '/' + image)
                labels.append(label)

    # load the model
    if model_name == 'batch':
        model.load_state_dict(torch.load('BatchCLIP.pt')['model_state_dict'])

    acc = zeroshot_train_fast(model, images, captions, labels, device)

    if model_name == 'batch':
        torch.save({'model_state_dict': model.state_dict(), }, 'BatchCLIP.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="cifar100",
                        choices=["cifar100-"], type=str)
    parser.add_argument('--model', default='batch',
                        choices=['batch', 'regular'], type=str)
    args = parser.parse_args()
    main(args)
