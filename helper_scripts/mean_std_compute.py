import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Path to the EuroSAT dataset
    data_dir = '../eurosat/train'

    # Load the dataset without transformations
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())

    # DataLoader to iterate over the dataset
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize variables to store the sums and squared sums
    mean = 0.0
    std = 0.0
    total_images = 0

    # Iterate through the dataset
    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size (the last batch can have fewer samples)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape (batch_size, 3, H * W)

        # Compute the mean and std for this batch
        mean += images.mean(2).sum(0)  # Sum over all pixels
        std += images.std(2).sum(0)  # Sum over all pixels
        total_images += batch_samples

    # Final mean and std across all images
    mean /= total_images
    std /= total_images

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")