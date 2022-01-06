from software.models.models_mnist import LeNet_P, LeNet_LL, LeNet_TWO_THIRD, LeNet_ALL
from software.models.models_cifar import ResNet_P, ResNet_LL, ResNet_ONE_THIRD, ResNet_TWO_THIRD, ResNet_HALF, ResNet_ALL
from software.models.models_svhn import VGG_P, VGG_LL, VGG_ONE_THIRD, VGG_TWO_THIRD, VGG_HALF, VGG_ALL

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model, input_size, output_size, args):
        net = None
        if model == "lenet_p":
            net = LeNet_P(input_size, output_size,  args)
        elif model == "lenet_ll":
            net = LeNet_LL(input_size, output_size, args)
        elif model == "lenet_two_third":
            net = LeNet_TWO_THIRD(input_size, output_size, args)
        elif model == "lenet_all":
            net = LeNet_ALL(input_size, output_size, args)
        elif model == "resnet_p":
            net = ResNet_P(input_size, output_size,  args)
        elif model == "resnet_ll":
            net = ResNet_LL(input_size, output_size, args)
        elif model == "resnet_one_third":
            net = ResNet_ONE_THIRD(input_size, output_size, args)
        elif model == "resnet_half":
            net = ResNet_HALF(input_size, output_size, args)
        elif model == "resnet_two_third":
            net = ResNet_TWO_THIRD(input_size, output_size, args)
        elif model == "resnet_all":
            net = ResNet_ALL(input_size, output_size, args)
        elif model == "vgg_p":
            net = VGG_P(input_size, output_size,  args)
        elif model == "vgg_ll":
            net = VGG_LL(input_size, output_size, args)
        elif model == "vgg_one_third":
            net = VGG_ONE_THIRD(input_size, output_size, args)
        elif model == "vgg_half":
            net = VGG_HALF(input_size, output_size, args)
        elif model == "vgg_two_third":
            net = VGG_TWO_THIRD(input_size, output_size, args)
        elif model == "vgg_all":
            net = VGG_ALL(input_size, output_size, args)
        else:
            raise NotImplementedError("Other models not implemented")

        return net
