'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
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
import clipscore
import json
from torchmetrics import Accuracy
import os
from pathlib import Path
from universal_mean import get_universal_embeddings

IMAGE_CAPTION_METRICS = Path('/share/cuvl/image_caption_metrics')
CIFAR100_DIR = '/share/cuvl/image_caption_metrics/cifar-100-python'
IMAGENET1K_DIR = '/share/cuvl/image_caption_metrics/imagenet-1k'
SUN397_DIR = '/share/cuvl/image_caption_metrics/SUN397'
STANFORDCARS_DIR = '/share/cuvl/image_caption_metrics/stanford_cars'


IMAGE_CAPTION_METRICS = Path('/share/cuvl/image_caption_metrics')
CIFAR100_DIR = '/share/cuvl/image_caption_metrics/cifar-100-python'
IMAGENET1K_DIR = '/share/cuvl/image_caption_metrics/imagenet-1k'
SUN397_DIR = '/share/cuvl/image_caption_metrics/SUN397'
STANFORDCARS_DIR = '/share/cuvl/image_caption_metrics/stanford_cars'


def zeroshot_prediction(model, images, captions, labels, device, args):
    model.eval().to(device)
    clip = model.clip
    clip.to(device)
    image_embeddings = clipscore.extract_all_images(
        images, clip, device, normalize=False).cpu()
    caption_embeddings = clipscore.extract_all_captions(
        captions, clip, device, normalize=False).cpu()
    if args.model == 'first':
        # image_embeddings = image_embeddings - \
        #     0.25*torch.mean(image_embeddings, dim=0)
        # caption_embeddings = caption_embeddings - \
        #     0.25*torch.mean(caption_embeddings, dim=0)
        # image_mean, text_mean = get_mscoco_mean(model, device)
        image_embeddings = image_embeddings - \
            torch.mean(image_embeddings,  dim=0).cpu()*0.25
        caption_embeddings = caption_embeddings - \
            torch.mean(caption_embeddings, dim=0).cpu() * 0.25
    image_embeddings = F.normalize(image_embeddings)
    caption_embeddings = F.normalize(caption_embeddings)
    num_classes = caption_embeddings.size(0)
    if isinstance(model, clipscore.OriginalCLIPScore) or isinstance(model, clipscore.FirstCLIPScore):
        sims = []
        for i in range(0, image_embeddings.size(0), 32):
            sims.append(image_embeddings[i:i+32] @ caption_embeddings.T)

    if isinstance(model, clipscore.BatchCLIP):
        sims = []
        for i in range(image_embeddings.size(0)):
            current_image = image_embeddings[i].broadcast_to(
                caption_embeddings.size())
            sims.append(model.train_forward(current_image,
                        caption_embeddings).reshape(1, -1))
    sims = torch.cat(sims, dim=0)
    preds = torch.argmax(sims, dim=1)
    # print(labels)
    # print(preds)
    acc = np.mean(preds.cpu().numpy() == np.array(labels))
    print('Using OriginalCLIPSore')
    print(f'Accuracy: {acc}')
    return acc


def zeroshot_train_fast(model, images, captions, labels, device,
                        max_iter=20, learning_rate=1e-2):
    """
    enter the training loop for max_iter iterations
    """
    model.train().to(device)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip.to(device)
    accuracy = Accuracy()
    print("=====> Computing Image Embeddings")
    image_features = clipscore.extract_all_images(images, clip, device).cpu()
    text_features = clipscore.extract_all_captions(
        captions, clip, device).cpu()
    data_loader = torch.utils.data.DataLoader(
        clipscore.CLIPJointDataset(image_features, labels),
        batch_size=64, num_workers=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.embedding_fc.parameters()] +
                                 [p for p in model.output_fc.parameters()] +
                                 [p for p in model.text_embedding_fc.parameters()], lr=learning_rate)
    embedding_size = image_features.size(1)
    num_classes = text_features.size(0)
    text = text_features.to(device).reshape(1, num_classes, -1)
    print("=====> Training")
    for i in tqdm.tqdm(range(max_iter)):
        losses = []
        accs = []
        for d in data_loader:
            images = d['image'].to(device)
            batch_size = images.size(0)
            images = images.reshape(batch_size, 1, -1)
            # caption actually here contains labels
            target = d['caption'].to(device).flatten()
            images = images.broadcast_to(
                (batch_size, num_classes, embedding_size))
            images = images.reshape(-1, embedding_size).to(device)
            batch_text = text[:].broadcast_to(
                (batch_size, num_classes, embedding_size))
            batch_text = batch_text.reshape(-1, embedding_size).to(device)
            pred = model.train_forward(
                images, batch_text).reshape(-1, num_classes)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(torch.argmax(pred, dim=1).cpu(), target.cpu())
            losses.append(loss.item())
            accs.append(acc.item())
        print(f"for iteration {i}: loss: {np.mean(losses)}")
        print(f"for iteration {i}: Acc: {np.mean(accs)}")
    return acc
