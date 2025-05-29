import os
import torch
import random
import numpy as np
import torch, torchvision
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import itertools
import config

from data.augmentation import center_crop_arr
from data.lsun_church import LSUNChurchesDataset
from data.lsun_bedroom import LSUNBedroomsDataset

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def load_dataset(args, batch_size=16):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.dataset_name == "ImageNet":
        pass
    elif args.dataset_name == "FFHQ":
        pass
    elif args.dataset_name == "CelebAHQ":
        pass
    elif args.dataset_name == "Bedrooms":
        pass
    elif args.dataset_name == "Churches":
        pass
    

def load_imagenet_dataset(data_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    val_dataset = ImageFolder(root=os.path.join(data_path, 'val'), transform=transform)
    len_val_set = len(val_dataset)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=False)
    return dataloader, len_val_set

def load_ffhq_dataset(data_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    val_dataset = ImageFolder(root=data_path, transform=eval_transform)
    len_val_set = len(val_dataset)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=False)
    return dataloader, len_val_set

def main():
    sample_folder_dir = "/projects/yuanai/processed_data/rFID/baselines/VAR"
    raw_folder_dir = "/projects/yuanai/processed_data/rFID/baselines/Input"

    ### load dataset
    data_path = "/projects/yuanai/data/ImageNet/"
    val_dataloader, len_val_set = load_dataset(data_path, batch_size=16)
    num_fid_samples = 50000

    ###load the vae checkpoint
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device='cuda', patch_nums= (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=1000, depth=16, shared_aln=False,
    )
    vae_ckpt = "/projects/yuanai/processed_data/checkpoint/VAR/vae_ch160v4096z32.pth"
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae = vae.cuda()
    vae.eval()

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    ssim, psnr, lpips = 0.0, 0.0, 0.0
    total = 0
    for idx, (x, _) in enumerate(val_dataloader):
        x = x.cuda()
        with torch.no_grad():
            x_rec = vae.img_to_reconstructed_img(x, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), last_one=True)
            batch_lpips = lpips_metric(x, x_rec).sum()

            samples = torch.clamp(127.5 * x_rec + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            input_samples = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(input_samples):
                index = i + total
                Image.fromarray(sample).save(f"{raw_folder_dir}/{index:06d}.png")

            total += 16

            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0

            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    print("PSNR:"+str(eval_psnr)+"  SSIM:"+str(eval_ssim)+ "  LPIPS:"+str(eval_lpips))

    create_npz_from_sample_folder(raw_folder_dir, num_fid_samples)

if __name__ == "__main__":
    main()