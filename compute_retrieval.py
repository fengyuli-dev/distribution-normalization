import clipscore
from clipscore import extract_all_captions, extract_all_images
from torchmetrics import Accuracy
import torch
import numpy as np
import torch.nn.functional as F
import argparse
import warnings
import json
from pathlib import Path
from dataset_paths import *


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn('Running on CPU.')

def main(args):
    refs = []
    images = []

    # load flickr30k dataset
    if args.dataset == 'flickr30k':
        dataset_path = '/share/cuvl/image_caption_metrics/flickr30k/flickr30k-images'
        with open('/share/cuvl/image_caption_metrics/flickr30k_test.txt', 'r') as fb:
            for line in fb:
                image = line.strip()
                images.append(dataset_path+'/'+image + '.jpg')
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
                refs.append(ref)

    # load mscoco dataset
    elif args.dataset == 'mscoco':
        with open('/share/cuvl/image_caption_metrics/MSCOCO_VAL2014/annotations/captions_val2014.json', 'r') as fb:
            caption_dicts = json.load(fb)['annotations']
        with open('/share/cuvl/image_caption_metrics/MSCOCO_VAL2014/annotations/coco_test_ids.npy', 'rb') as fb:
            test_ids = set(np.load(fb))
        image2caption = {}
        for d in caption_dicts:
            image = d['image_id']
            if not d['id'] in test_ids:
                continue
            if not image in image2caption:
                image2caption[image] = []
            cap = d['caption'].strip().split(' ')
            cap = ' '.join(cap)
            image2caption[image].append(cap)
        for image, captions in image2caption.items():
            images.append('/share/cuvl/image_caption_metrics/MSCOCO_VAL2014/val2014/COCO_val2014_' +
                          str(image).rjust(12, '0')+'.jpg')
            refs.append(captions)

    # load model
    if args.model == 'dn':
        model = clipscore.DNCLIPScore()
    elif args.model == 'regular':
        model = clipscore.OriginalCLIPScore()
    model.to(device)

    # TODO
    model.clip = torch.load(f'{ckpt_name}.pt')

    print('====> Doing Retrieval')
    compute_retrieval(model, images, refs, device)

def compute_retrieval(model, images, refs, device, verbose=True):
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
    # img_labels is the corresponding image of the ref
    img_labels = []
    txt_labels = {}
    for i, rs in enumerate(refs):
        txt_labels[i] = []
        for j, r in enumerate(rs):
            all_refs.append(r)
            img_labels.append(i)
            txt_labels[i].append(i*len(rs) + j)

    image_features = extract_all_images(images, model.clip, device).to(device)
    text_features = extract_all_captions(
        all_refs, model.clip, device).to(device)
    if isinstance(model, clipscore.DNCLIPScore):
        image_features = image_features - 0.25 * \
            torch.mean(image_features, dim=0)
        text_features = text_features - 0.25*torch.mean(text_features, dim=0)

    img_labels = torch.Tensor(img_labels).long()
    
    # Text->Images retrieval
    t2i_sim = (text_features @ image_features.T).cpu()
    num_classes_img = t2i_sim.size(1)
    i_top1 = Accuracy(top_k=1, task="multiclass",
                      num_classes=num_classes_img)(t2i_sim, img_labels)
    i_top5 = Accuracy(top_k=5, task="multiclass",
                      num_classes=num_classes_img)(t2i_sim, img_labels)
    i_top10 = Accuracy(top_k=10, task="multiclass",
                       num_classes=num_classes_img)(t2i_sim, img_labels)
    t2i_results = [i_top1, i_top5, i_top10]
    
    # Images->Text retrieval
    sim = t2i_sim.T
    indexes = torch.argsort(sim, dim=1, descending=True)[:, :10]
    w, h = indexes.size()
    index_labels = torch.zeros(w, h).long()
    for i in range(w):
        for j in range(h):
            index_labels[i, j] = img_labels[indexes[i, j]]
    t_top1 = torch.mean(torch.where(
        torch.sum(index_labels[:, :1] == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
    t_top5 = torch.mean(torch.where(
        torch.sum(index_labels[:, :5] == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
    t_top10 = torch.mean(torch.where(
        torch.sum(index_labels == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
    i2t_results = [t_top1, t_top5, t_top10]
    
    if verbose:
        print(f'Top-1 Accuracy (Text->Image): {i_top1}')
        print(f'Top-5 Accuracy (Text->Image): {i_top5}')
        print(f'Top-10 Accuracy (Text->Image): {i_top10}')
        print(f'Top-1 Accuracy (Image->Text): {t_top1}')
        print(f'Top-5 Accuracy (Image->Text): {t_top5}')
        print(f'Top-10 Accuracy (Image->Text): {t_top10}')

    return i2t_results, t2i_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="flickr30k", choices=["flickr30k", "mscoco"], type=str)
    parser.add_argument('--model', default='dn', choices=['regular', 'dn'], type=str)
    args = parser.parse_args()
    main(args)