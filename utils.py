from torchvision import transforms
from torch.autograd import Variable,grad
import torch


class config:
    units=[32,64,128,256,512,1024]
    padding=[2,2,2,2,2]
    k_size=[5,5,5,5,5]#kernal size
    strides=[2,2,2,2,2]
    fs=[8,8]
    image_size=[3,128,128]
    small_image_size=[3,57,57]

    
    batch_size = 64 if torch.cuda.is_available() else 4
    latent_dim = 120
    imsize = 128

    save_path = 'result/'
    imgs_path = 'images/'

    train_augmentation = transforms.Compose(
        [
            transforms.Resize(size = (128,128)),
            transforms.ToTensor()
        ]
                                )

    @staticmethod
    def weights_init(m):#wgan need carefully initialized weights.
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    @staticmethod
    def get_noise(batch_size=32):
        return Variable(torch.rand(batch_size,config.latent_dim))




