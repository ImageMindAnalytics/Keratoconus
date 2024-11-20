import os
import torch
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import ParameterSampler
from torchreg import AffineRegistration
from matplotlib.widgets import Slider
import math
from typing import Optional, Tuple
import pandas as pd 
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

def nmi_gauss(x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of similar shape"

    def gaussian_window(x, bins, sigma):
        assert x.ndim == 2, "Input tensor should be 2-dimensional."
        return torch.exp(
            -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
        ) / (math.sqrt(2 * math.pi) * sigma)

    x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
    x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
    p_XY = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
    p_XY += e  # deal with numerical instability

    p_XY = p_XY / p_XY.sum((1, 2))[:, None, None]

    p_X = p_XY.sum(1)
    p_Y = p_XY.sum(2)

    I = (p_XY * torch.log(p_XY / (p_X[:, None] * p_Y[:, :, None]))).sum((1, 2))

    marg_ent_0 = (p_X * torch.log(p_X)).sum(1)
    marg_ent_1 = (p_Y * torch.log(p_Y)).sum(1)

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


def nmi_gauss_mask(x1, x2, x1_bins, x2_bins, mask, sigma=1e-3, e=1e-10):
    def gaussian_window_mask(x, bins, sigma):

        assert x.ndim == 1, "Input tensor should be 2-dimensional."
        return torch.exp(-((x[None, :] - bins[:, None]) ** 2) / (2 * sigma ** 2)) / (
            math.sqrt(2 * math.pi) * sigma
        )

    x1_windowed = gaussian_window_mask(torch.masked_select(x1, mask), x1_bins, sigma)
    x2_windowed = gaussian_window_mask(torch.masked_select(x2, mask), x2_bins, sigma)
    p_XY = torch.mm(x1_windowed, x2_windowed.transpose(0, 1))
    p_XY = p_XY + e  # deal with numerical instability

    p_XY = p_XY / p_XY.sum()

    p_X = p_XY.sum(0)
    p_Y = p_XY.sum(1)

    I = (p_XY * torch.log(p_XY / (p_X[None] * p_Y[:, None]))).sum()

    marg_ent_0 = (p_X * torch.log(p_X)).sum()
    marg_ent_1 = (p_Y * torch.log(p_Y)).sum()

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


class NMI(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(self,intensity_range: Optional[Tuple[float, float]] = None,nbins: int = 32,sigma: float = 0.1,use_mask: bool = False,):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(fixed_range[0],fixed_range[1],self.nbins,dtype=fixed.dtype,device=fixed.device,)
        bins_warped = torch.linspace(warped_range[0],warped_range[1],self.nbins,dtype=fixed.dtype,device=fixed.device,)

        return -nmi_gauss(fixed, warped, bins_fixed, bins_warped, sigma=self.sigma).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(fixed_range[0],fixed_range[1],self.nbins,dtype=fixed.dtype,device=fixed.device,)
        bins_warped = torch.linspace(warped_range[0],warped_range[1],self.nbins,dtype=fixed.dtype,device=fixed.device,)

        return -nmi_gauss_mask(fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma)


# Function to pad an image to the largest dimensions (z, y, x)
def pad_image_to_largest(image, largest_shape):
    # Get the numpy array from the image
    image_array = sitk.GetArrayFromImage(image)
    
    # Calculate padding amounts for each dimension
    pad_y = (largest_shape[0] - image_array.shape[0]) // 2
    pad_x = (largest_shape[1] - image_array.shape[1]) // 2

    # Pad the array to match the largest dimensions
    padded_array = np.pad(image_array,
                          ((pad_y, largest_shape[0] - image_array.shape[0] - pad_y), (pad_x, largest_shape[1] - image_array.shape[1] - pad_x)),
                          mode='constant', constant_values=0)

    # Convert the padded array back to a SimpleITK image
    padded_image = sitk.GetImageFromArray(padded_array)
    
    # Manually copy the origin, spacing, and direction from the original image
    padded_image.SetOrigin(image.GetOrigin())
    padded_image.SetSpacing(image.GetSpacing())
    padded_image.SetDirection(image.GetDirection())
    
    return padded_image

# Function to find the largest dimensions for each axis (z, y, x)
def find_largest_dimensions(images_with_paths):
    # Get the shape of each image (in the order: z, y, x)
    image_shapes = [sitk.GetArrayFromImage(img).shape for _, img in images_with_paths]
    
    # Find the largest size in each axis (z, y, x)
    max_y = max(shape[0] for shape in image_shapes)
    max_x = max(shape[1] for shape in image_shapes)
    
    return (max_y, max_x)

# Function to pad all images to the largest dimensions
def pad_all_images_to_largest(images_with_paths):
    # Find the largest dimensions for each axis
    largest_shape = find_largest_dimensions(images_with_paths)
    
    # Pad all images to match the largest dimensions (z, y, x) and keep track of their paths
    padded_images_with_paths = [(path, pad_image_to_largest(img, largest_shape)) for path, img in images_with_paths]
    
    return padded_images_with_paths

# Function to compute the voxel-wise mean of cropped images while keeping track of paths
def compute_mean_image(cropped_images):
    # Convert all cropped images to numpy arrays and compute the mean
    try:
        mean_image = np.mean([sitk.GetArrayFromImage(img) for _, img in cropped_images], axis=0)
    except:
        mean_image = np.mean([sitk.GetArrayFromImage(img) for img in cropped_images], axis=0)

    # Convert the mean numpy array back to a SimpleITK image and copy the metadata from the first image
    mean_image_itk = sitk.GetImageFromArray(mean_image)

    if isinstance(cropped_images[0], tuple):  # Handle both cases where it's (path, img) or just img
        reference_image = cropped_images[1][1]
    else:
        reference_image = cropped_images[0]

    mean_image_itk.SetOrigin(reference_image.GetOrigin())
    mean_image_itk.SetSpacing(reference_image.GetSpacing())
    mean_image_itk.SetDirection(reference_image.GetDirection())

    return mean_image_itk

# Function to load all images in the folder and normalize them
def load_images(folder, csv):
    images = []
    df = pd.read_csv(csv)
    for idx, row in df.iterrows():
        file_path = os.path.join(folder, row['path'])
        img = sitk.ReadImage(file_path)
        norm_img = normalize_image(img)
        images.append((file_path, norm_img))
    return images

# Function to normalize the image intensities to the range [0, 1]
def normalize_image(image):
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    # convert to grayscale 
    image_array = (0.2126*image_array[:,:,0] + 0.7152*image_array[:,:,1] + 0.0722*image_array[:,:,2])

    norm_img = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    # isVector=True to indicate the last dimension represents vector components (channels), not a spatial dimension.
    norm_img = sitk.GetImageFromArray(norm_img)
    if norm_img.GetDimension() != image.GetDimension():
        raise ValueError(f"Dimension mismatch: original image {image.GetDimension()}D, normalized image {norm_img.GetDimension()}D")


    # Copy the origin, spacing, and direction from the original image
    norm_img.SetOrigin(image.GetOrigin())
    norm_img.SetSpacing(image.GetSpacing())
    norm_img.SetDirection(image.GetDirection())

    return norm_img
def dice_loss(x1, x2):
    dim = [2, 3, 4] if len(x2.shape) == 5 else [0, 1]
    inter = torch.sum(x1 * x2, dim=dim)
    union = torch.sum(x1 + x2, dim=dim)
    return 1 - (2. * inter / union).mean()

def registration(mean, input_img, image_path, iter, patient_id, output_folder, param_sampler):
    # Save the registered image
    image_folder = os.path.join(output_folder, f'iter_{iter}')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_path = os.path.join(image_folder, patient_id.replace('.png', 'registered.jpg'))
    
    best_loss = 0.0  # Initialize to track the best loss
    best_image = input_img
    # for params in param_sampler:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n\033[1mIteration: {iter} -- Using {device.upper()} -- Registering : {image_path} with mean\033[0m")

    moving = torch.from_numpy(sitk.GetArrayFromImage(input_img)).float().to(device)
    static = torch.from_numpy(sitk.GetArrayFromImage(mean)).float().to(device)

    # Normalize the images
    epsilon = 1e-8    # Small value to avoid division by zero
    moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + epsilon)
    static_normed = (static - static.min()) / (static.max() - static.min() + epsilon)

    # Initialize NMI loss function for rigid registration
    nmi_loss_function_rigid = NMI(intensity_range=None, nbins=64, sigma=0.05, use_mask=False)

    # Initialize AffineRegistration for Rigid registration
    reg_rigid = AffineRegistration(scales=(4, 2), iterations=(100, 30), is_3d=False,
                                   dissimilarity_function=nmi_loss_function_rigid.metric, 
                                    # dissimilarity_function=dice_loss, 
                                    optimizer=torch.optim.Adam,
                                    with_translation=True, with_rotation=True, with_zoom=False, with_shear=False,
                                    align_corners=True, padding_mode='zeros')
    
    # Perform rigid registration
    moved_image = reg_rigid(moving_normed[None, None], static_normed[None, None])[0, 0]

    # Compute the final loss
    final_loss = dice_loss(moved_image, static_normed)
    print(f"Final Loss (NMI): {final_loss}")

    image = sitk.GetImageFromArray(moved_image.cpu().numpy())

    image.SetOrigin(mean.GetOrigin())
    image.SetSpacing(mean.GetSpacing())
    image.SetDirection(mean.GetDirection())

    if final_loss > best_loss and final_loss > 1e-5:
        best_loss = final_loss
        print(f"New best parameters found with loss: {best_loss}")
        
        best_image = image
        # sitk.WriteImage(image, image_path)
        image = sitk.Cast(255*image, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_path)
        writer.UseCompressionOn()
        writer.Execute(image)

    return best_image




# Main function to iterate and compute the final average
def main(args, param_sampler):
    # Generate output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Load all images from the folder
    images = load_images(args.img_folder, args.csv)

    # Pad all images to the size of the largest volume
    padded_images = pad_all_images_to_largest(images)

    # Compute the mean of the padded images
    mean_image = 255*compute_mean_image(padded_images)
    mean_path = os.path.join(args.output_folder, 'mean_image.jpg')
    mean_image = sitk.Cast(mean_image, sitk.sitkUInt8)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(mean_path)
    writer.UseCompressionOn()
    writer.Execute(mean_image)

    for iteration in range(args.num_iterations):
        # Register all images to the current mean
        registered_images = []
        for path, img in padded_images:
            patient_id = os.path.basename(path)
            registered_image = registration(mean_image, img, path, iteration, patient_id, args.output_folder, param_sampler)
            registered_images.append((path, registered_image))

        # Update padded_images with the newly registered images
        padded_images = registered_images
        
        # Compute the new mean from the registered images
        mean_image = 255*compute_mean_image(registered_images)

        # Save the final mean image
        mean_path.replace('.jpg', f"iter_{iteration}.jpg")
        mean_image = sitk.Cast(mean_image, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(mean_path)
        writer.UseCompressionOn()
        writer.Execute(mean_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--csv', type=str, required=True, help='csv containing file names')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where the mean will be saved')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations for iterative mean computation')
    
    args = parser.parse_args()

    # Define the parameter grid for hyperparameter search
    param_grid = {
        'learning_rate_rigid': np.logspace(-5, -3, 5),   # Learning rate for rigid registration
        'sigma_rigid': np.logspace(-2, -1, 3),            # Number of iterations for rigid registration
    }

    # Number of parameter combinations to sample
    n_samples = 15
    param_sampler = ParameterSampler(param_grid, n_iter=n_samples)

    main(args, param_sampler)


