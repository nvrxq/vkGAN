import torch
from yaml import parse
from models import G_net
from utils import config
import argparse
from torchvision.transforms import ToPILImage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_to_generate', type=int, default=50)
    parser.add_argument('--path_to_save', type = str, default = 'generated_img/')
    args = parser.parse_args()
    use_GPU = torch.cuda.is_available()
    device = torch.device('cuda' if use_GPU else 'cpu')
    print(device)
    # d_model.load_state_dict(torch.load('/home/nvrxq/deepVK/weights/d_model.pth'),map_location=torch.device('cpu'))
    g_model = G_net(config.latent_dim)
    checkpoint = torch.load("./weights/generator.pth", map_location=torch.device('cpu'))
    g_model.load_state_dict(checkpoint)
    g_model.eval()
    noise = config.get_noise(args.len_to_generate)
    print('Start')
    with torch.no_grad():
        generated = g_model(noise).cpu()
    for item in enumerate(generated):
            img = ToPILImage()(item[1])
            img.save(f'./{args.path_to_save}/{item[0]}.png')
