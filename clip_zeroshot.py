import argparse
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPModel
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.functional import cosine_similarity
import torch
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
import scipy
import random
import os
import clip_score
import json
from torchmetrics import Accuracy
import os


# the coefficient for distribution normalization
LAMBDA = 0.25


def zeroshot_prediction(model, images, val_images, captions, labels, device, args):
    model.eval().to(device)
    clip = model.clip
    clip.to(device)
    image_embeddings = clip.extract_all_images(
        images, clip, device, normalize=False).cpu()
    caption_embeddings = clip.extract_all_captions(
        captions, clip, device, normalize=False).cpu()
    top1s = []
    top5s = []
    for _ in range(5):
        _image_embeddings = image_embeddings
        _caption_embeddings = caption_embeddings
        if args.model == 'first':
            val_embeddings = clip.extract_all_images(
                random.sample(val_images, args.num_samples), clip, device, normalize=False).cpu()
            _image_embeddings = image_embeddings - \
                torch.mean(val_embeddings, dim=0).cpu()*LAMBDA
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
