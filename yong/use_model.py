import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# Custom DataLoader
import matplotlib.image as mpimg
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

# model
from torchvision import models
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

from PIL import Image 
NUM_CLASS = 8


class Resize(object):
    def __call__(self, sample):
        if type(sample) is dict:
            image, target = sample['image'], sample['target']
            return {'image': cv2.resize(image, (224, 224)),
                    'target': cv2.resize(target, (224, 224))}
        else:
            return cv2.resize(sample, (224, 224))

class GrayScale(object):
    def __call__(self, sample):
        if type(sample) is dict:
            image, target = sample['image'], sample['target']
            target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            return {'image': image, 'target': target}
        else:
            return sample
    
class Normalization(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        if type(sample) is dict:
            image, target = sample['image'], sample['target']

            image = image.astype(np.float64) 
            image /= 255.0
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

            target = target.astype(np.float64) 
    #         target /= 255.0

            return {'image': image, 'target': target}
        else:
            image = sample
            image = image.astype(np.float64) 
            image /= 255.0
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
            return image

class Discretization(object):
    def __init__(self, num_class):
        self.num_class = num_class
    
    def __call__(self, sample):
        if type(sample) is dict:
            image, target = sample['image'], sample['target']
            bins = np.linspace(0, 256, self.num_class)
            target = np.digitize(target, bins)
            return {'image': image, 'target': target}
        else:
            return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        if type(sample) is dict:
            image, target = sample['image'], sample['target']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'target': torch.from_numpy(target)}
        else:
            image = sample.transpose((2, 0, 1))
            return torch.from_numpy(image)


class TransformData(object):
	def __init__(self, *args):
		self.data_transforms = transforms.Compose(args)

	def transform_data(self, data):
		return self.data_transforms(data)



def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

# TODO: can be moved into the class FCN8s if necessary
def load_pretrained_FCN8(save_file, for_eval=True):
    netfcn8 = FCN8s(n_class=NUM_CLASS)
    netfcn8.load_state_dict(torch.load(save_file, map_location='cpu'))
    if for_eval:
        netfcn8.eval()
    print("Loaded FCN8 Successfully")
    return netfcn8

# TODO: can be moved into the class FCN8s if necessary
# TODO: batch prediction
def predict_attention(net, transform_data, input_image, output_folder):
    """
    :param net: Neural network ready for prediction
                (input: (batch size, channels, height, width))
                (output: (batch size, labels, height, width))
    :param transform_data:
    :param input_image: image filename
    :param output_folder: folder to store the attention map
    """

    # load and transform the input image
    input_image = pathlib.Path(input_image)
    im = cv2.imread(str(input_image.absolute()))
    height, width, _ = im.shape
    im = transform_data.transform_data(im)
    new_im = im.numpy()
    new_im = torch.tensor(new_im[np.newaxis, ...], requires_grad=False).float()  # add batch size dimension
    # prediction
    pred = net.forward(new_im).detach().numpy()

    # convert prediction into a 2D attention map
    pred = pred.transpose(0, 2, 3, 1)
    converted = np.zeros((224, 224))
    for i in range(224):
        for j in range(224):
            converted[i, j] = np.argmax(pred[0, i, j])

    # resize the attention map
    resized_converted = cv2.resize(converted, (width, height))

    # save the attention map
    resized_converted = resized_converted / (np.max(resized_converted) - np.min(resized_converted)) * 255.0
    cv2.imwrite(f"{output_folder}/output_{input_image.name}", resized_converted)


if __name__ == "__main__":
    net = load_pretrained_FCN8("fcn8s_attn_pred.pkt")

    transform_data = TransformData(Resize(),
                                   GrayScale(),
                                   Normalization(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5)),
                                   Discretization(NUM_CLASS),
                                   ToTensor())

    predict_attention(net,
                      transform_data,
                      "/Users/zhengxinyong/Desktop/AILAB190/graphics/5690086434_7f2996aec5_b.jpg",
                      "/Users/zhengxinyong/Desktop/AILAB190/")

    
    