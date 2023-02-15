import clipscore
from clipscore import extract_all_captions, extract_all_images
from torchmetrics import Accuracy
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path


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
    t2i_results = [t_top1, t_top5, t_top10]
    
    # Images->Text retrieval
    sim = t2i_sim.T
    indexes = torch.argsort(sim, dim=1, descending=True)[:, :10]
    w, h = indexes.size()
    index_labels = torch.zeros(w, h).long()
    for i in range(w):
        for j in range(h):
            index_labels[i, j] = img_labels[indexes[i, j]]
    i_top1 = torch.mean(torch.where(
        torch.sum(index_labels[:, :1] == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
    i_top5 = torch.mean(torch.where(
        torch.sum(index_labels[:, :5] == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
    i_top10 = torch.mean(torch.where(
        torch.sum(index_labels == torch.arange(w).reshape(-1, 1), dim=1) > 0, 1.0, 0.0))
    i2t_results = [i_top1, i_top5, i_top10]
    
    if verbose:
        print(f'Top-1 Accuracy (Text->Image): {i_top1}')
        print(f'Top-5 Accuracy (Text->Image): {i_top5}')
        print(f'Top-10 Accuracy (Text->Image): {i_top10}')
        print(f'Top-1 Accuracy (Image->Text): {t_top1}')
        print(f'Top-5 Accuracy (Image->Text): {t_top5}')
        print(f'Top-10 Accuracy (Image->Text): {t_top10}')

    return i2t_results, t2i_results
