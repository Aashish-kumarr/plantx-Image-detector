from src.models.model1_resnet9 import ResNet9Model1

def create_model2_14class_resnet9():
    """
    Returns a fresh randomly initialized ResNet-9 model configured for 14 crop classes.
    """
    return ResNet9Model1(in_channels=3, num_classes=14)
