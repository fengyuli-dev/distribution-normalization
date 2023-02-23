'''
CLIP and CLIP + DN models
'''
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from utils import *
from torch.nn.functional import cosine_similarity
from transformers import CLIPModel
LAMBDA = .25


class OriginalCLIPScore(nn.Module):

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    def forward(self, images, text):
        image_features = self.clip.get_image_features(images)
        text_features = self.clip.get_text_features(text)
        similarity = cosine_similarity(image_features, text_features)
        return 2.5 * torch.maximum(similarity, torch.zeros_like(similarity))


class DNCLIPScore(nn.Module):

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.image_constant = torch.zeros(512,)
        self.text_constant = torch.zeros(512,)
        self.tau = 1

    def forward(self, images, text):
        image_features = self.clip.get_image_features(images)
        text_features = self.clip.get_text_features(text)
        image_features = F.normalize(image_features)
        text_features = F.normalize(text_features)
        image_features = image_features - LAMBDA*self.image_constant
        text_features = text_features - LAMBDA*self.text_constant
        similarity = torch.sum(image_features * text_features, dim=1)
        return similarity


def get_clip_score(model, images, captions, device, refs=None):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(model, DNCLIPScore):
        image_features = torch.Tensor(extract_all_images(
            images, model.clip, device, num_workers=1)).cpu()
        model.image_constant = torch.mean(image_features, dim=0).to(device)
        all_refs = []
        for ref in refs:
            all_refs = all_refs + ref
        text_features = torch.Tensor(extract_all_captions(
            all_refs, model.clip, device, num_workers=1)).cpu()
        model.text_constant = torch.mean(text_features, dim=0).to(device)
    image_data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=64, num_workers=1, shuffle=False)
    text_data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=64, num_workers=1, shuffle=False)
    per = []
    for images, text in tqdm.tqdm(zip(image_data, text_data)):
        images = images['image'].to(device)
        text = text['caption'].to(device)
        with torch.no_grad():
            per.append(model(images, text).cpu().numpy().flatten())
    per = np.concatenate(per, axis=0)
    return np.mean(per), per, captions


def get_clip_score_ref(model, images, captions, references, device):
    '''
    get reference-based image-text clipscore variants.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(model, DNCLIPScore):
        image_features = torch.Tensor(extract_all_images(
            images, model.clip, device, num_workers=1)).cpu()
        model.image_constant = torch.mean(image_features, dim=0).to(device)
        text_features = torch.Tensor(extract_all_captions(
            captions, model.clip, device, num_workers=1)).cpu()
        model.text_constant = torch.mean(text_features, dim=0).to(device)

    image_data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=64, num_workers=1, shuffle=False)
    text_data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=64, num_workers=1, shuffle=False)
    per = []
    for images, text in tqdm.tqdm(zip(image_data, text_data)):
        images = images['image'].to(device)
        text = text['caption'].to(device)
        with torch.no_grad():
            per.append(model(images, text).cpu().numpy().flatten())
    per = np.concatenate(per, axis=0)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])
    flattened_refs = extract_all_captions(
        flattened_refs, model.clip, device).cpu().numpy()
    caption_embeddings = extract_all_captions(
        captions, model.clip, device).cpu().numpy()
    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}
    text_per = []
    for c_idx, cand in tqdm.tqdm(enumerate(caption_embeddings)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        text_per.append(np.max(all_sims))

    per = np.array(per)
    text_per = np.array(text_per)
    if isinstance(model, DNCLIPScore):
        per = per + text_per
    else:
        per = 2 * per * text_per / (per + text_per)

    return np.mean(per), per, captions


def get_clip_score_pascal(model, device, get_ref=False, gt_refs=None):
    ds = Pascal50sDataset()
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=64, num_workers=1, shuffle=True)
    if isinstance(model, DNCLIPScore):
        images = [os.path.join(os.path.join(
            ds.root, "images"), d[0][0]) for d in ds.data]
        image_features = torch.Tensor(extract_all_images(
            images, model.clip, device, num_workers=1)).cpu()
        model.image_constant = torch.mean(image_features, dim=0).to(device)
        if refs:
            refs_flattened = [
                single_ref for ref_list in refs for single_ref in ref_list]
            text_features = torch.Tensor(extract_all_captions(
                gt_refs, model.clip, device, num_workers=1)).cpu()
        else:
            captions = [c[1][0] for c in ds.data]+[c[2][0]
                                                   for c in ds.data]
            text_features = torch.Tensor(extract_all_captions(
                captions, model.clip, device, num_workers=1)).cpu()
        model.text_constant = torch.mean(text_features, dim=0).to(device)

    # idx_to_cat = {1: "hc", 2: "hi", 3: "hm", 4: "mm"}
    category_to_num_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    category_to_num_total = {1: 0, 2: 0, 3: 0, 4: 0}

    for images, a_captions, b_captions, references, categories, labels in tqdm.tqdm(data_loader, maxinterval=len(data_loader)):
        images = images.to(device)
        a_captions = a_captions.to(device)
        b_captions = b_captions.to(device)
        with torch.no_grad():
            a_scores = model(images, a_captions).cpu().numpy().flatten()
            b_scores = model(images, b_captions).cpu().numpy().flatten()

        if get_ref:
            flattened_refs = []
            flattened_refs_idxs = []
            for refs in references:
                flattened_refs.extend(refs)
                flattened_refs_idxs.extend([idx for idx in range(len(refs))])
            flattened_refs = extract_all_captions(
                flattened_refs, model.clip, device).cpu().numpy()

            cand_idx2refs = collections.defaultdict(list)
            for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
                cand_idx2refs[cand_idx].append(ref_feats)
            cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

            a_caption_embeddings = extract_all_captions_tokenized(
                a_captions, model.clip)
            b_caption_embeddings = extract_all_captions_tokenized(
                b_captions, model.clip)
            a_text_per, b_text_per = [], []
            for idx, caption_embeddings in enumerate([a_caption_embeddings, b_caption_embeddings]):
                for c_idx, cand in enumerate(caption_embeddings):
                    cur_refs = cand_idx2refs[c_idx]
                    cand = cand.cpu().numpy()
                    all_sims = cand.dot(cur_refs.transpose())
                    if idx == 0:
                        a_text_per.append(np.max(all_sims))
                    else:
                        b_text_per.append(np.max(all_sims))

            a_text_per, b_text_per = np.array(a_text_per), np.array(b_text_per)
            if isinstance(model, DNCLIPScore):
                a_scores = a_scores + a_text_per
                b_scores = b_scores + b_text_per
            else:
                a_scores = 2 * a_scores * a_text_per / (a_scores + a_text_per)
                b_scores = 2 * b_scores * b_text_per / (b_scores + b_text_per)

        preds = np.array(
            [1 if a_score < b_score else 0 for a_score, b_score in zip(a_scores, b_scores)])

        categories = categories.cpu().numpy()
        for category, pred, label in zip(categories, preds, labels):
            category_to_num_correct[category] += 1 if pred == label else 0
            category_to_num_total[category] += 1

    hc_acc = category_to_num_correct[1]/category_to_num_total[1]
    hi_acc = category_to_num_correct[2]/category_to_num_total[2]
    hm_acc = category_to_num_correct[3]/category_to_num_total[3]
    mm_acc = category_to_num_correct[4]/category_to_num_total[4]
    mean = (hc_acc+hi_acc+hm_acc+mm_acc)/4

    return hc_acc, hi_acc, hm_acc, mm_acc, mean
