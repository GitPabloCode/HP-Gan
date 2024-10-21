import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake):
    batch_size, number_seq, joints, xyz = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, number_seq, joints, xyz)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images.view(interpolated_images.shape[0], -1))

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty