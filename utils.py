'''
PyTorch dataset for experiments
'''
import os
import random

import clip
import scipy
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)


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
        c_data = clip_score.tokenize(
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
