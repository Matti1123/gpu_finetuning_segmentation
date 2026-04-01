import segmentation_models_pytorch as smp


def build_unet_resnet34(encoder_weights="imagenet"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model