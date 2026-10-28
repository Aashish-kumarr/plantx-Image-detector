from src.models.model1_resnet9 import ResNet9Model1

def create_model2_resnet9():
    """
    Returns a fresh randomly initialized ResNet-9 model configured for Model 2.
    Output: 3 classes (Potato, Tomato, Other)
    """
    return ResNet9Model1(in_channels=3, num_classes=3)
