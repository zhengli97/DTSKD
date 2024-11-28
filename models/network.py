#--------------
#CNN-architecture
#--------------
from torch.types import Number
from models import *

from models.resnet_dtskd import resnet18_dtskd
from models.resnext2_dtskd import resnext18_dtskd
from models.vgg_dtskd import vgg16_dtskd
from models.shufflenetv2_dtskd import shufflenetv2

from utils.color import Colorer

C = Colorer.instance()

def get_network(args):
    if args.data_type == 'cifar100':
        if args.classifier_type == 'resnet18_dtskd':
            net = resnet18_dtskd(num_classes=100)
        elif args.classfier_type == 'resnext18_dtskd':
            net = resnext18_dtskd(num_classes=100)
        elif args.classifier_type == 'shufflev2_dtskd':
            net = shufflenetv2(num_classes=100)
        elif args.classifier_type == 'vgg16_dtskd':
            net = vgg16_dtskd(num_classes=100)

    if args.data_type == 'imagenet':
        if args.classifier_type == 'ResNet152':
            net = ResNet(dataset = 'imagenet', depth=152, num_classes=1000, bottleneck=True)

    print(C.underline(C.yellow("[Info] Building model: {}".format(args.classifier_type))))

    return net