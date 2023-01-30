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
import collections
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'candidates_json',
        type=str,
        help='Candidates json mapping from image_id --> candidate.')

    parser.add_argument(
        'image_dir',
        type=str,
        help='Directory of images, with the filenames as image ids.')

    parser.add_argument(
        '--references_json',
        default=None,
        help='Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default=1,
        type=int,
        help='If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, human_score=None, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '
        self.human_score = human_score

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        if self.human_score is not None:
            return {'caption': c_data, 'human_score': self.human_score[idx]}
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class CLIPImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, data, captions, human_scores):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)
        self.captions = captions
        self.prefix = 'A photo depicts '
        self.human_scores = human_scores

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        c_data = clip.tokenize(
            self.prefix + self.captions[idx], truncate=True).squeeze()
        return {'image': image, 'caption': c_data, 'human_score': self.human_scores[idx]}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=1, normalize=True):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in data:
            b = b['caption'].to(device)
            features = model.get_text_features(b)
            if normalize:
                features = F.normalize(features, p=2, dim=1)
            all_text_features.append(features)
    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features


def get_caption_mean(captions, model, device, batch_size=256, num_workers=1):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    caption_mean = torch.zeros(512,).to(device)
    count = 0
    with torch.no_grad():
        for b in data:
            b = b['caption'].to(device)
            caption_mean += torch.sum(
                model.get_text_features(b), dim=0)
            count += b.size(0)
    return caption_mean/count


def get_image_mean(images, model, device, batch_size=64, num_workers=1):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    image_mean = torch.zeros(512,).to(device)
    count = 0
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            image_mean += torch.sum(model.get_image_features(b), dim=0)
            count += b.size(0)
    return image_mean/count


def extract_all_captions_tokenized(captions, model):
    all_text_features = []
    with torch.no_grad():
        all_text_features.append(F.normalize(
            model.get_text_features(captions)))
    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=1, normalize=True):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            features = model.get_image_features(b)
            if normalize:
                features = F.normalize(features, p=2, dim=1)
            all_image_features.append(features)
    all_image_features = torch.cat(all_image_features, dim=0)
    return all_image_features


class BatchCLIP(nn.Module):
    def __init__(self,
                 batch_size: int = 1,
                 device="cuda") -> None:
        super(BatchCLIP, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.embedding_fc = orthogonal(nn.Linear(512, batch_size, bias=False))
        self.text_embedding_fc = orthogonal(
            nn.Linear(512, batch_size, bias=False))
        self.batch_size = batch_size
        self.output_fc = nn.Sequential(nn.Linear(
            batch_size*2+1, 1, bias=True))
        self.sf = nn.Softmax(dim=-1)
        self.device = device

    def train_forward(self, images, text):
        image_features = images
        text_features = text
        sim = torch.sum(image_features * text_features, dim=1).reshape(-1, 1)
        batch_sim = self.embedding_fc(image_features)
        batch_sim_text = self.text_embedding_fc(text_features)
        sf_sim = torch.cat([sim, batch_sim, batch_sim_text], dim=1)
        sf_sim = self.output_fc(sf_sim)
        return sf_sim

    def forward(self, images, text):
        image_features = self.clip.get_image_features(images)
        text_features = self.clip.get_text_features(text)
        image_features = F.normalize(image_features)
        text_features = F.normalize(text_features)
        sim = torch.sum(image_features * text_features, dim=1).reshape(-1, 1)
        batch_sim = self.embedding_fc(image_features)
        batch_sim_text = self.text_embedding_fc(text_features)
        sf_sim = torch.cat([sim, batch_sim, batch_sim_text], dim=1)
        sf_sim = self.output_fc(sf_sim)
        return sf_sim


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


class FirstCLIPScore(nn.Module):

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
        similarity = cosine_similarity(image_features, text_features)
        image_cali = torch.sum(text_features * self.image_constant, dim=1)
        text_cali = torch.sum(image_features*self.text_constant, dim=1)
        return torch.exp((similarity - image_cali)/self.tau) + torch.exp((similarity - text_cali)/self.tau)

    def forward_fast(self, image_features, text_features):
        similarity = cosine_similarity(image_features, text_features)
        image_cali = torch.sum(text_features * self.image_constant, dim=1)
        text_cali = torch.sum(image_features*self.text_constant, dim=1)
        return torch.exp((similarity - image_cali)/self.tau) + torch.exp((similarity - text_cali)/self.tau)


def get_clip_score(model, images, captions, device, refs=None):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    # '''
    if isinstance(model, FirstCLIPScore):
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


def get_full_clip_score(model, images, captions, device):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    # '''
    image_features = torch.Tensor(extract_all_images(
        images, model.clip, device, num_workers=1)).to(device)
    text_features = torch.Tensor(extract_all_captions(
        captions, model.clip, device, num_workers=1)).to(device)
    # text_features = text_features - torch.mean(text_features, dim=0)
    # print(image_features.size())

    num_samples = len(image_features)
    similarities = torch.sum(
        image_features * text_features, dim=1)
    image_text = image_features @ text_features.T

    similarities = similarities.reshape(num_samples, 1)
    print((similarities - image_text).size())
    per = torch.mean(torch.exp(similarities - image_text), dim=1)
    per = per + torch.mean(torch.exp(similarities - image_text.T), dim=1)
    # per = 0 - per
    per = per.cpu().numpy()
    return np.mean(per), per, captions


def get_clip_score_ref(model, images, captions, references, device):
    '''
    get reference-based image-text clipscore variants.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    # '''
    if isinstance(model, FirstCLIPScore):
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
    if isinstance(model, FirstCLIPScore):
        per = per + text_per
    else:
        per = 2 * per * text_per / (per + text_per)

    return np.mean(per), per, captions


def train_clip_score(model, images, captions, human_scores, device,
                     max_iter=20, learning_rate=2e-5):
    # image_data = torch.utils.data.DataLoader(
    #     CLIPImageDataset(images),
    #     batch_size=64, num_workers=1, shuffle=False)
    # text_data = torch.utils.data.DataLoader(
    #     CLIPCapDataset(captions, human_scores),
    #     batch_size=64, num_workers=1, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        CLIPImageCaptionDataset(images, captions, human_scores),
        batch_size=64, num_workers=1, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([p for p in model.embedding_fc.parameters()] +
                                 [p for p in model.output_fc.parameters()] +
                                 [p for p in model.text_embedding_fc.parameters()], lr=learning_rate)
    for i in tqdm.tqdm(range(max_iter)):
        losses = []
        for d in data_loader:
            images = d['image'].to(device)
            text = d['caption'].to(device)
            scores = d['human_score'].to(device).float().flatten()
            pred = model(images, text).flatten()
            loss = criterion(pred, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"for iteration {i}: loss: {np.mean(losses)}")
    return model


class CLIPJointDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions, human_scores=None, prefix='A photo depicts'):
        self.images = images
        self.captions = captions
        self.human_scores = None
        if human_scores is not None:
            self.human_scores = human_scores
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        if self.human_scores is not None:
            return {'caption': self.captions[idx], 'image': self.images[idx],
                    'human_score': self.human_scores[idx]}
        else:
            return {'caption': self.captions[idx], 'image': self.images[idx]}

    def __len__(self):
        return len(self.images)


def train_clip_score_fast(model, images, captions, human_scores, device,
                          max_iter=20, learning_rate=1e-1):
    """
    Faster training loop by precomputing clip features
    """
    image_features = torch.Tensor(extract_all_images(
        images, model.clip, device, num_workers=1)).cpu()
    text_features = torch.Tensor(extract_all_captions(
        captions, model.clip, device, num_workers=1)).cpu()
    human_scores = torch.Tensor(human_scores).reshape(-1, 1).cpu()
    # device = "cpu"
    # model.to(device)
    data_loader = torch.utils.data.DataLoader(
        CLIPJointDataset(image_features, text_features, human_scores),
        batch_size=64, num_workers=1, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([p for p in model.embedding_fc.parameters()] +
                                 [p for p in model.output_fc.parameters()] +
                                 [p for p in model.text_embedding_fc.parameters()], lr=learning_rate)
    for i in tqdm.tqdm(range(max_iter)):
        losses = []
        for d in data_loader:
            images = d['image'].to(device)
            text = d['caption'].to(device)
            scores = d['human_score'].to(device).float().flatten()
            pred = model.train_forward(images, text).flatten()
            loss = criterion(pred, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # wandb.log({'loss':np.mean(losses)})
        print(f"for iteration {i}: loss: {np.mean(losses)}")
    return model


class Pascal50sDataset(torch.utils.data.Dataset):
    idx2cat = {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}

    def __init__(self,
                 root: str = "/share/cuvl/image_caption_metrics/pascal",
                 media_size: int = 224):
        super().__init__()
        self.root = root
        self.read_data(self.root)
        self.read_score(self.root)
        self.transform = self._transform_test()
        self.prefix = "A photo depicts"

    def _transform_test(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    @staticmethod
    def loadmat(path):
        return scipy.io.loadmat(path)

    def read_data(self, root):
        mat = self.loadmat(
            os.path.join(root, "pyCIDErConsensus/pair_pascal.mat"))
        self.data = mat["new_input"][0]
        self.categories = mat["category"][0]
        # sanity check
        c = torch.Tensor(mat["new_data"])
        hc = (c.sum(dim=-1) == 12).int()
        hi = (c.sum(dim=-1) == 13).int()
        hm = ((c < 6).sum(dim=-1) == 1).int()
        mm = ((c < 6).sum(dim=-1) == 2).int()
        assert 1000 == hc.sum()
        assert 1000 == hi.sum()
        assert 1000 == hm.sum()
        assert 1000 == mm.sum()
        assert (hc + hi + hm + mm).sum() == self.categories.shape[0]
        chk = (torch.Tensor(self.categories) - hc - hi * 2 - hm * 3 - mm * 4)
        assert 0 == chk.abs().sum(), chk

    def read_score(self, root):
        mat = self.loadmat(
            os.path.join(root, "pyCIDErConsensus/consensus_pascal.mat"))
        data = mat["triplets"][0]
        # self.gt_refs = list(set([triplet[0][0][0][0] for triplet in data]))
        # data contains reference + candidate captions
        # triplets[0][0] is reference
        # triplets[0][0] is candidate 1
        # triplets[0][0] is candidate 2
        # self.data contains only candidate captions with image name
        self.gt_refs = []
        self.labels = []
        self.references = []
        for i in range(len(self)):
            votes = {}
            refs = []
            for j in range(i * 48, (i + 1) * 48):
                a, b, c, d = [x[0][0] for x in data[j]]
                key = b[0].strip() if 1 == d else c[0].strip()
                refs.append(a[0].strip())
                votes[key] = votes.get(key, 0) + 1
            # simulate "random selection of 5 ground-truth references from 48 candidate"
            self.gt_refs += refs[:5]
            assert 2 >= len(votes.keys()), votes
            assert len(votes.keys()) > 0
            try:
                vote_a = votes.get(self.data[i][1][0].strip(), 0)
                vote_b = votes.get(self.data[i][2][0].strip(), 0)
            except KeyError:
                print("warning: data mismatch!")
                print(f"a: {self.data[i][1][0].strip()}")
                print(f"b: {self.data[i][2][0].strip()}")
                print(votes)
                exit()
            # Ties are broken randomly.
            label = 0 if vote_a > vote_b + random.random() - .5 else 1
            self.labels.append(label)
            self.references.append(refs)

    def __len__(self):
        return len(self.data)

    def get_image(self, filename: str):
        path = os.path.join(self.root, "images")
        img = Image.open(os.path.join(path, filename)).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx: int):
        vid, a, b = [x[0] for x in self.data[idx]]
        label = self.labels[idx]
        feat = self.get_image(vid)
        a = clip.tokenize(self.prefix + a.strip(), truncate=True).squeeze()
        b = clip.tokenize(self.prefix + b.strip(), truncate=True).squeeze()
        references = self.references[idx]
        category = self.categories[idx]
        return feat, a, b, references, category, label


def get_clip_score_pascal(model, device, get_ref=False, gt_refs=None):
    ds = Pascal50sDataset()
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=64, num_workers=1, shuffle=True)
    if isinstance(model, FirstCLIPScore):
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
            if isinstance(model, FirstCLIPScore):
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
