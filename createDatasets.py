import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

gaussian_noise_levels       = [2.5, 5, 10, 15, 22.5]
salt_pepper_noise_levels    = [0.03, 0.06, 0.12, 0.18, 0.27]
poisson_brightness_scales   = [0.8, 0.6, 0.4, 0.2, 0.1]
seeds                       = [0, 1, 2, 3, 4]

input_dir  = Path(r"C:\Users\Admin\source\repos\COCO YOLO Validation\val\images")
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

def add_gaussian_noise(img, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise     = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, prob, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    noisy = img.copy()
    black, white = 0, 255

    probs = rng.random(img.shape[:2])

    noisy[probs < (prob / 2)] = black
    noisy[probs > 1 - (prob / 2)] = white

    return noisy

def add_poisson_gaussian_noise(
    image: np.ndarray,
    exposure_scale: float = 0.8,
    scale: float = 255.0,
    sigma_read: float = 5.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float32) / 255.0
    darkened = img * exposure_scale
    scaled = darkened * scale
    poisson_sampled = rng.poisson(scaled).astype(np.float32)
    poisson_rescaled = poisson_sampled / scale
    noisy_uint8_float = poisson_rescaled * 255.0
    noisy_with_read_noise = noisy_uint8_float + rng.normal(0.0, sigma_read, size=noisy_uint8_float.shape)
    noisy_clipped = np.clip(noisy_with_read_noise, 0, 255).astype(np.uint8)

    return noisy_clipped

def process_images(
    do_gaussian=True,
    do_salt_pepper=True,
    do_poisson_gaussian=True
):
    images = glob(str(input_dir / "*"))

    for seed in seeds:
        rng = np.random.default_rng(seed)

        if do_gaussian:
            for i, sigma in enumerate(gaussian_noise_levels):
                folder = output_dir / f"gaussian/level_{i+1}/seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)

                for img_path in tqdm(images, desc=f"Gaussian σ={sigma}, seed={seed}"):
                    out_path = folder / Path(img_path).name
                    if out_path.exists():
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    noisy = add_gaussian_noise(img, sigma, rng=rng)
                    cv2.imwrite(str(out_path), noisy)

        if do_salt_pepper:
            for i, p in enumerate(salt_pepper_noise_levels):
                folder = output_dir / f"salt_pepper/level_{i+1}/seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)

                for img_path in tqdm(images, desc=f"S&P p={p}, seed={seed}"):
                    out_path = folder / Path(img_path).name
                    if out_path.exists():
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    noisy = add_salt_pepper_noise(img, p, rng=rng)
                    cv2.imwrite(str(out_path), noisy)

        if do_poisson_gaussian:
            for b_idx, brightness_scale in enumerate(poisson_brightness_scales):
                for g_idx, sigma in enumerate(gaussian_noise_levels):
                    folder = (
                        output_dir / 
                        f"poisson_gaussian/brightness_{b_idx+1}/gaussian_{g_idx+1}/seed_{seed}"
                    )
                    folder.mkdir(parents=True, exist_ok=True)

                    for img_path in tqdm(images, desc=f"PG b={brightness_scale}, σ={sigma}, seed={seed}"):
                        out_path = folder / Path(img_path).name
                        if out_path.exists():
                            continue
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        noisy = add_poisson_gaussian_noise(
                            img,
                            exposure_scale=brightness_scale,
                            sigma_read=sigma,
                            rng=rng
                        )
                        cv2.imwrite(str(out_path), noisy)

if __name__ == "__main__":
    process_images(do_gaussian=True, do_salt_pepper=True, do_poisson_gaussian=True)