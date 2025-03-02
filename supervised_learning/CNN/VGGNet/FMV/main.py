import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from torchinfo import summary
import PIL.Image as Image

import matplotlib.pyplot as plt

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = vgg16(weights=VGG16_Weights.DEFAULT)
model = model.to(DEVICE)

summary(model=model, input_size=(1, 3, 224, 224))


def viz(layer_num, img, model, device):
    # transforms_vgg16 = VGG16_Weights.IMAGENET1K_V1.transforms
    transforms_vgg16 = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                   std=(0.229, 0.224, 0.225))])

    img = transforms_vgg16(img).unsqueeze(dim=0)
    img = img.to(device)

    layer = model.features[layer_num]
    print(layer)

    # define hook to return output feature map at layer n
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # register the hook
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.inference_mode():
        preds = model(img)

    pred_cls = preds.argmax(dim=1)
    classes_imgnet_txt = r"C:\Users\User\Desktop\Projects\OGP\open-genome-project\supervised_learning\CNN\VGGNet\imagenetv1.txt"
    with open(classes_imgnet_txt, 'r') as f:
        classes_imgnet = f.readlines()
    pred_cls = classes_imgnet[pred_cls]
    print(pred_cls)

    layer_output = feature_maps[0].squeeze()
    rows, cols = 4, 6
    fig = plt.figure(figsize=(10, 6))
    for i in range(1, (rows * cols) + 1):
        feature_map = layer_output[i - 1, :, :].cpu().numpy()
        fig.add_subplot(rows, cols, i)
        plt.imshow(feature_map, cmap='viridis')
        plt.tight_layout()
        plt.axis(False)

    plt.show()

# img_noise = transforms.ToPILImage()(torch.randn(3, 224, 224))

# viz(10, img_noise, model, DEVICE)

img_car = Image.open("C:/Users/User/Desktop/Projects/OGP/open-genome-project/supervised_learning/CNN/VGGNet/car_01.png").convert('RGB')

viz(10, img_car, model, DEVICE) 