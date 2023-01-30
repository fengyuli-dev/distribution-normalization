import argparse
import pickle
import torch
import clipscore
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

cc12m_tar_dir = Path("/share/cuvl/image_caption_metrics/cc12m/cc12m")
cc12m_img_dir = Path(cc12m_tar_dir, "cc12m_images")
cc12m_caption_dir = Path(cc12m_tar_dir, "cc12m_captions")
scalar_save_path = Path(Path.cwd(), "universal_means", "scalar.p")
vector_normalized_path = Path(Path.cwd(), "universal_means", "vector_normalized.p")
vector_not_normalized_save_path = Path(Path.cwd(), "universal_means", "vector_not_normalized.p")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.type == "scalar":
        save_path = scalar_save_path
    elif args.type == "vector":
        if args.normalize:
            save_path = vector_normalized_path
        else:
            save_path = vector_not_normalized_save_path
    if save_path.exists():
        save_path.unlink()

    print(f"Saving to {save_path}")

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip = clip.to(device)

    type_to_embedding = {}

    images = []
    for img_file in tqdm(cc12m_img_dir.glob("*.jpg"), unit_scale=True):
        images.append(img_file)
    image_embeddings = clipscore.extract_all_images(images, clip, device, normalize=args.normalize)
    if args.type == "scalar":
        image_embeddings_mean = torch.mean(image_embeddings)
    elif args.type == "vector":
        image_embeddings_mean = torch.mean(image_embeddings, dim=0)
    type_to_embedding["image_embedding"] = image_embeddings_mean.detach().clone()
    del images
    del image_embeddings
    del image_embeddings_mean

    captions = []
    for txt_file in tqdm(cc12m_caption_dir.glob("*.txt"), unit_scale=True):
        caption = txt_file.read_text()
        captions.append(caption)
    caption_embeddings = clipscore.extract_all_captions(captions, clip, device)
    if args.type == "scalar":
        caption_embeddings_mean = torch.mean(caption_embeddings)
    elif args.type == "vector":
        caption_embeddings_mean = torch.mean(caption_embeddings, dim=0)
    type_to_embedding["txt_embedding"] = caption_embeddings_mean.detach().clone()
    del captions
    del caption_embeddings
    del caption_embeddings_mean

    pickle.dump(type_to_embedding, open(str(save_path), "wb"))
    print(f"Dumped to {str(save_path)}")


def get_universal_embeddings(mean_type="vector", normalize=False):
    if mean_type == "scalar":
        save_path = str(scalar_save_path)
    elif mean_type == "vector":
        if normalize:
            save_path = vector_normalized_path
        else:
            save_path = vector_not_normalized_save_path
    type_to_embedding = pickle.load((open(save_path, "rb")))
    return type_to_embedding["image_embedding"], type_to_embedding["txt_embedding"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="vector",
                        choices=["scalar", "vector"], type=str)
    parser.add_argument("--normalize", action="store_true", default=False)
    main(parser.parse_args())
