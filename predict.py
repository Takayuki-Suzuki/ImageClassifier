from get_input_args import get_prediction_args
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
import json

def main():
    in_arg = get_prediction_args()
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda' if in_arg.gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Start predicting on GPU...")
    else:
        print("Start predicting on CPU...")
        
    model = load_model_from_checkpoint(in_arg.checkpoint_directory + '/checkpoint.pth')
    image = process_image(in_arg.image_path, device).unsqueeze(0)
        
    model.to(device)
    model.eval()
    with torch.no_grad():
        image.to(device)
        logps = model(image)
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(in_arg.top_k, dim=1)
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
        titles = [cat_to_name[str(i)] for i in top_class.data.cpu().numpy()[0]]
    probabilities = top_ps.data.cpu().numpy()[0],
    print(titles)

def process_image(image, device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    width, height = image.size
    resized_width, resized_height = 0, 0
    if width >= height:
        resized_height = 256
        resized_width = width * resized_height // height
    else:
        resized_width = 256
        resized_height = width * resized_width // width
    image = image.resize((resized_width, resized_height))
    width, height = image.size
    image = image.crop(((width - 224) // 2,
                         (height - 224) // 2,
                         (width + 224) // 2,
                         (height + 224) // 2))
    np_image = np.array(image)/256.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1)).squeeze()
    if device == 'cuda':
        return torch.from_numpy(np_image).type(torch.cuda.FloatTensor)
    else:
        return torch.from_numpy(np_image).type(torch.FloatTensor)

def load_model_from_checkpoint(file_path):
    checkpoint= torch.load(file_path, map_location=lambda storage, loc: storage)
    model= getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
#     print(model.class_to_idx)
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
#     optimizer.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model

if __name__ == "__main__":
    main()