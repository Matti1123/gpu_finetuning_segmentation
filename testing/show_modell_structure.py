import torch
from torchinfo import summary

from models.U_net_resnet34 import build_unet_resnet34


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_resnet34(encoder_weights="imagenet").to(device)

    summary(
        model,
        input_size=(1, 3, 256, 256),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=4,
    )


if __name__ == "__main__":
    main()