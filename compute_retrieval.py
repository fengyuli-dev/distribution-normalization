import clip_score
from utils import *
from torchmetrics import Accuracy
import torch
import numpy as np
import torch.nn.functional as F
import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

import clipscore
from clipscore import extract_all_captions, extract_all_images
from dataset_paths import *

# the coefficient for distribution normalization used in our paper
LAMBDA = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn('Running on CPU.')

def main(args):
    refs = []
    images = []
    dev_images = []
    dev_refs = []

    # load flickr30k dataset
    if args.dataset == 'flickr30k':
        dataset_path = Path(FLICKR30K_DIR, "flickr30k-images")
        sentences_path = Path(IMAGE_CAPTION_METRICS, "flickr30k_sentences")
        image_paths = os.listdir(dataset_path)
        all_images = [i.replace('.jpg', '')
                      for i in image_paths if i.endswith('.jpg')]
        # with open('/share/cuvl/image_caption_metrics/flickr30k_test.txt', 'r') as fb:
        with open(Path(IMAGE_CAPTION_METRICS, 'flickr30k_test.txt'), 'r') as fb:
            for line in fb:
                image = line.strip()
                image_path = Path(dataset_path, f"{image}.jpg")
                images.append(str(image_path))
                ref_path = Path(sentences_path, f"{image}.txt")
                # ref_path = '/share/cuvl/image_caption_metrics/flickr30k_sentences/' + image + '.txt'
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
                refs.append(ref)
        for image in all_images:
            dev_image_path = str(Path(dataset_path, f"{image}.jpg"))
            if dev_image_path in images:
                continue
            dev_images.append(dev_image_path)
            ref_path = Path(sentences_path, f"{image}.txt")
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
            dev_refs.append(ref)
        assert len(dev_images) + len(images) == len(all_images)

    # load mscoco dataset
    elif args.dataset == 'mscoco':
        with open(Path(MSCOCO_DIR, "annotations", "captions_val2014.json"), "r") as fb:
            caption_dicts = json.load(fb)['annotations']
        with open(Path(MSCOCO_DIR, "annotations", "coco_test_ids.npy"), "rb") as fb:
            test_ids = set(np.load(fb))
        with open(Path(MSCOCO_DIR, "annotations", "coco_dev_ids.npy"), "rb") as fb:
            dev_ids = set(np.load(fb))
        image2caption = {}
        dev_image2caption = {}
        for d in caption_dicts:
            image = d['image_id']
            if not d['id'] in test_ids:
                continue
            if not image in image2caption:
                image2caption[image] = []
            cap = d['caption'].strip().split(' ')
            cap = ' '.join(cap)
            image2caption[image].append(cap)

        for d in caption_dicts:
            image = d['image_id']
            if not d['id'] in dev_ids:
                continue
            if not image in dev_image2caption:
                dev_image2caption[image] = []
            cap = d['caption'].strip().split(' ')
            cap = ' '.join(cap)
            dev_image2caption[image].append(cap)

        for image, captions in image2caption.items():
            img_path = Path(MSCOCO_DIR, "val2014", f"COCO_val2014_{str(image).rjust(12, '0')}.jpg")
            images.append(str(img_path))
            refs.append(captions)
        for image, captions in dev_image2caption.items():
            dev_img_path = Path(MSCOCO_DIR, "val2014", f"COCO_val2014_{str(image).rjust(12, '0')}.jpg")
            dev_images.append(str(dev_img_path))
            dev_refs.append(captions)

    # load model
    if args.dn:
        model = clipscore.DNCLIPScore()
    else:
        model = clipscore.OriginalCLIPScore()

    print('====> Doing Retrieval')
    compute_retrieval(model, images, refs, dev_images, dev_refs, device, args)

def compute_retrieval(model, images, refs, dev_images, dev_refs, device, args):
    unique_images = []
    unique_refs = []
    saved = set()
    for image, ref in zip(images, refs):
        if not image in saved:
            unique_images.append(image)
            unique_refs.append(ref)
            saved.add(image)
    images = unique_images
    refs = unique_refs
    all_refs = []
    # labels is the corresponding image of the ref
    labels = []
    for i, rs in enumerate(refs):
        for r in rs:
            all_refs.append(r)
            labels.append(i)
    dev_all_refs = []
    for _, rs in enumerate(dev_refs):
        for r in rs:
            dev_all_refs.append(r)
    image_features = extract_all_images(images, model.clip, device).cpu()
    text_features = extract_all_captions(all_refs, model.clip, device).cpu()
    labels = torch.Tensor(labels).long()
    if args.image_to_text:
        print("Image to Text")
    else:
        print("Text to Image")


    top1s = []
    top5s = []
    top10s = []
    for _ in range(args.num_experiments):
        _image_features = image_features
        _text_features = text_features
        # image_means, text_means = get_mscoco_mean(model, device)
        if args.dn:
            dev_image_features = extract_all_images(random.sample(
                dev_images, args.num_samples), model.clip, device).cpu()
            dev_text_features = extract_all_captions(random.sample(
                dev_all_refs, args.num_samples), model.clip, device).cpu()
            _image_features = _image_features - LAMBDA * \
                torch.mean(dev_image_features, dim=0)
            _text_features = _text_features - LAMBDA * \
                torch.mean(dev_text_features, dim=0)

        sim = (_text_features @ _image_features.T).cpu()
        if args.image_to_text:
            sim = sim.T
            indexes = torch.argsort(sim, dim=1, descending=True)[:, :10]
            w, h = indexes.size()
            index_labels = torch.zeros(w, h).long()
            for i in range(w):
                for j in range(h):
                    index_labels[i, j] = labels[indexes[i, j]]
            top1 = torch.mean(torch.where(
                torch.sum(index_labels[:, :1] == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
            top5 = torch.mean(torch.where(
                torch.sum(index_labels[:, :5] == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
            top10 = torch.mean(torch.where(
                torch.sum(index_labels == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
        else:
            num_classes_img = sim.size(1)
            top1 = Accuracy(top_k=1, task="multiclass",
                      num_classes=num_classes_img)(sim, labels)
            top5 = Accuracy(top_k=5, task="multiclass",
                      num_classes=num_classes_img)(sim, labels)
            top10 = Accuracy(top_k=10, task="multiclass",
                       num_classes=num_classes_img)(sim, labels)
        top1s.append(top1*100)
        top5s.append(top5*100)
        top10s.append(top10*100)

    print(f'Top-1 Accuracy: {np.mean(top1s)}')
    print(f'Top-1 Std {np.std(top1s)}')
    print(f'Top-5 Accuracy: {np.mean(top5s)}')
    print(f'Top-5 Std {np.std(top5s)}')
    print(f'Top-10 Accuracy: {np.mean(top10s)}')
    print(f'Top-10 Std {np.std(top10s)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="flickr30k", choices=["flickr30k", "mscoco"], type=str)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--num_experiments', default=5, type=int)
    parser.add_argument('--dn', action="store_true", default=False)
    retrieval_task = parser.add_argument_group("retrieval_task")
    retrieval_task.add_argument('--image_to_text', action="store_true", default=True)
    retrieval_task.add_argument('--text_to_image', action="store_true", default=False)
    args = parser.parse_args()
    main(args)