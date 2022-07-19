import cv2
import math
import numpy as np

import torch
import torchvision.transforms as transforms

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def compute_psnr(images, labels):

    batch, _, _, _ = images.size()
    PSNR = 0
    for i in range(batch):
        PSNR += calculate_psnr(images[i], labels[i])

    PSNR = PSNR / batch

    return PSNR

def calculate_psnr(img1, img2, border=0):

    # img1 and img2 have range [0, 255]
    img1 = imgtoimg(img1)
    img2 = imgtoimg(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(images, labels):

    batch, _, _, _ = images.size()
    SSIM = 0
    for i in range(batch):

        SSIM += calculate_ssim(images[i], labels[i])

    SSIM = SSIM / batch
    return SSIM


def calculate_ssim(img1, img2, border=0):

    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = imgtoimg(img1)
    img2 = imgtoimg(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def compute_bef(img):
    block = 8
    height, width = img.shape[:2]

    H = [i for i in range(width - 1)]
    H_B = [i for i in range(block - 1, width - 1, block)]
    H_BC = list(set(H) - set(H_B))

    V = [i for i in range(height - 1)]
    V_B = [i for i in range(block - 1, height - 1, block)]
    V_BC = list(set(V) - set(V_B))

    D_B = 0
    D_BC = 0

    for i in H_B:
        diff = img[:, i] - img[:, i + 1]
        D_B += np.sum(diff ** 2)

    for i in H_BC:
        diff = img[:, i] - img[:, i + 1]
        D_BC += np.sum(diff ** 2)

    for j in V_B:
        diff = img[j, :] - img[j + 1, :]
        D_B += np.sum(diff ** 2)

    for j in V_BC:
        diff = img[j, :] - img[j + 1, :]
        D_BC += np.sum(diff ** 2)

    N_HB = height * (width / block - 1)
    N_HBC = height * (width - 1) - N_HB
    N_VB = width * (height / block - 1)
    N_VBC = width * (height - 1) - N_VB
    D_B = D_B / (N_HB + N_VB)
    D_BC = D_BC / (N_HBC + N_VBC)
    eta = math.log2(block) / math.log2(min(height, width)) if D_B > D_BC else 0
    return eta * (D_B - D_BC)


# --------------------------------------------
def calculate_psnrb(img1, img2, border=0):
	# img1: ground truth
	# img2: compressed image
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    img1 = imgtoimg(img1)
    img2 = imgtoimg(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    if img2.shape[-1]==3:
        img2_y = rgb2ycbcr(img2).astype(np.float64)
        bef = compute_bef(img2_y)
    else:
        img2 = img2.astype(np.float64)
        bef = compute_bef(img2)
    mse = np.mean((img1 - img2)**2)
    mse_b = mse + bef
    if mse_b == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse_b))


def compute_psnrb(labels, images):

    batch, _, _, _ = images.size()
    PSNRB = 0
    for i in range(batch):
        PSNRB += calculate_psnrb(labels[i], images[i])

    PSNRB = PSNRB / batch

    return PSNRB

def imgtoimg(img):

    img = img.data.float().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.uint8((img.clip(0, 1) * 255.).round())

    return img


def get_val_psnr(img, label):

    h, w, c = img.shape
    h_len = h // 3
    w_len = w // 3

    img = imgtoimg(img)
    label = imgtoimg(label)

    img = img.astype(np.float64)
    label = label.astype(np.float64)

    list_psnr = []
    for i in range(3):

        h_start = i * h_len

        for j in range(3):
            w_start = j * w_len

            if i != 2 and j != 2:
                img_patch = img[:, h_start:h_start + h_len, w_start:w_start + w_len]
                label_patch = label[:, h_start:h_start + h_len, w_start:w_start + w_len]
                sum = (img_patch - label_patch) ** 2
                patch_sq = np.sum(sum)
                list_psnr.append(patch_sq)

            if i == 2 and j != 2:
                img_patch = img[:, h_start:, w_start:w_start + w_len]
                label_patch = label[:, h_start:, w_start:w_start + w_len]
                patch_sq = np.sum((img_patch - label_patch) ** 2)
                list_psnr.append(patch_sq)

            if i != 2 and j == 2:
                img_patch = img[:, h_start:h_start + h_len, w_start:]
                label_patch = label[:, h_start:h_start + h_len, w_start:]
                patch_sq = np.sum((img_patch - label_patch) ** 2)
                list_psnr.append(patch_sq)

            if i == 2 and j == 2:
                img_patch = img[:, h_start:, w_start:]
                label_patch = label[:, h_start:, w_start:]
                patch_sq = np.sum((img_patch - label_patch) ** 2)
                list_psnr.append(patch_sq)

    all = 0
    for k in range(9):
        all += list_psnr[k]
    mse = all / (h * w * c)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def compute_val_psnr(images, labels):

    batch, _, _, _ = images.size()
    PSNR = 0
    for i in range(batch):
        PSNR += get_val_psnr(images[i], labels[i])

    PSNR = PSNR / batch

    return PSNR


def get_sum_mse(list_out, list_label, h, w, c):

    all_mse = 0
    for i in range(len(list_out)):
        img_patch = imgtoimg(list_out[i][0])
        label_patch = imgtoimg(list_label[i][0])

        img_patch = img_patch.astype(np.float64)
        label_patch = label_patch.astype(np.float64)

        patch_mse = np.sum((img_patch - label_patch) ** 2)
        all_mse += patch_mse

    avg_mse = all_mse / (h * w * c)

    return avg_mse


def compute_val_PSNR(list_out, list_label, h, w, c):

    mse = get_sum_mse(list_out, list_label, h, w, c)

    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))



'''
def imgto(img):

    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    return img

transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

img1 = cv2.imread('G:\dataset\LIVE1\RGB\JPEG\Q10/bikes.jpg')
img2 = cv2.imread('G:\dataset\LIVE1\RGB/0_PNG_GT/bikes.png')
img1, img2 = imgto(img1), imgto(img2)
img1, img2 = transform(img1), transform(img2)
img1, img2 = torch.unsqueeze(img1, dim=0), torch.unsqueeze(img2, dim=0)


P = compute_psnr(img1, img2)
print('PSNR', P)

psnr = compute_val_psnr(img1, img2)
print(psnr)

si = compute_ssim(img1, img2)
print(si)
'''