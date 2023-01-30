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
import random
from clip_retrieval import compute_retrieval
from clipscore import Pascal50sDataset, CLIPImageCaptionDataset
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import CLIPModel
from slack_logging import progress_alerts
import tqdm


IMAGE_CAPTION_METRICS = Path('/share/cuvl/image_caption_metrics')
FLICKR8K_DIR = Path(IMAGE_CAPTION_METRICS, 'flickr8k')
FLICKR30K_DIR = Path(IMAGE_CAPTION_METRICS, 'flickr30k')
MSCOCO_DIR = Path(IMAGE_CAPTION_METRICS, 'MSCOCO_VAL2014')
COMPOSITE_DIR = Path(IMAGE_CAPTION_METRICS, 'composite')
PASCAL_DIR = Path(IMAGE_CAPTION_METRICS, 'pascal')


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn('Running on CPU.')

@progress_alerts(func_name="train_retrieval.py")
def main(args):
    images = []
    captions = []
    ckpt_name = None

    if args.dataset == "mscoco":
        ckpt_name = 'cocoCLIP'
        print("Fine tuning on MSCOCO")
        with open('/share/cuvl/image_caption_metrics/MSCOCO_VAL2014/annotations/captions_val2014.json', 'r') as fb:
            caption_dicts = json.load(fb)['annotations']
        image2caption = {}
        for d in caption_dicts:
            image = d['image_id']
            if not image in image2caption:
                image2caption[image] = []
            cap = d['caption'].strip().split(' ')
            cap = ' '.join(cap)
            image2caption[image].append(cap)
        for image, _captions in image2caption.items():
            image_path = '/share/cuvl/image_caption_metrics/MSCOCO_VAL2014/val2014/COCO_val2014_' +\
                str(image).rjust(12, '0')+'.jpg'
            for caption in _captions:
                images.append(image_path)
                captions.append(caption)
    elif args.dataset == "flickr30k":
        ckpt_name = 'flickr30Clip'
        ckpt_save_path = Path(Path.cwd(), "flickr_finetune_4")
        if not ckpt_save_path.exists():
            ckpt_save_path.mkdir()

        print(f"Fine tuning on Flickr30k, saving to {ckpt_save_path}")
        dataset_path = '/share/cuvl/image_caption_metrics/flickr30k/flickr30k-images'
        with open('/share/cuvl/image_caption_metrics/flickr30k_test.txt', 'r') as fb:
            for line in fb:
                image = line.strip()
                img_path = (dataset_path+'/'+image + '.jpg')
                ref_path = '/share/cuvl/image_caption_metrics/flickr30k_sentences/' + image + '.txt'
                ref = []
                with open(ref_path, 'r') as f2:
                    for raw in f2:
                        splitted = raw.split(' ')
                        processed = []
                        for s in splitted:
                            if '[' in s:
                                continue
                            else:
                                processed.append(
                                    s.replace(']', '').replace('\n', ''))
                        ref.append(' '.join(processed))
                for caption in ref:
                    images.append(img_path)
                    captions.append(caption)

    # dummy human_scores
    human_scores = [0 for _ in images]
    accelerator = Accelerator()
    device = accelerator.device
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.02)
    model.train()
    data_loader = torch.utils.data.DataLoader(
        CLIPImageCaptionDataset(images, captions, human_scores),
        batch_size=64, num_workers=4, shuffle=True)
    model, data_loader = accelerator.prepare(model, data_loader)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(args.max_iter):
        losses = []
        for d in tqdm.tqdm(data_loader):
            images = d['image'].to(device)
            text = d['caption'].to(device)
            image_features = F.normalize(model.get_image_features(images))
            text_features = F.normalize(model.get_text_features(text))
            sim = (image_features @ text_features.T) * \
                np.exp(args.temperature)
            labels = torch.arange(sim.size(0)).to(device)
            # loss = -torch.sum(torch.diagonal(sim /
            #                   torch.sum(sim, dim=1).reshape(-1, 1)))-torch.sum(torch.diagonal(sim /
            #                                                                                   torch.sum(sim, dim=0).reshape(1, -1)))
            loss = criterion(sim, labels) + criterion(sim.T, labels)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            losses.append(loss.item())
        print(f"for iteration {i}: loss: {np.mean(losses)}")
        torch.save(model, str(Path(ckpt_save_path, f"{ckpt_name}_epoch_{i}.pt")))
    print(f"Saved to {ckpt_name=}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="flickr8k-expert", choices=["flickr8k-expert",
                                                                         "thumb", "pascal", "composite", "flickr8k-cf", "flickr30k", "mscoco"], type=str)
    parser.add_argument('--max_iter', default=5, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    args = parser.parse_args()
    main(args)
