from get_input_args import get_training_args
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import workspace_utils
from collections import OrderedDict

def main():
    in_arg = get_training_args()
    
    print(in_arg.data_directory)
    print(in_arg.arch)
    print(in_arg.epochs)
    print(in_arg.hidden_units)
    print(in_arg.learning_rate)
    print(in_arg.save_dir)
    print(in_arg.gpu)
    
    data_dir = in_arg.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    
    model = getattr(models, in_arg.arch)(pretrained=True)
    
    input_size = model.classifier[0].in_features
    output_size = 102
    
    for param in model.parameters():
        param.requires_grad = False

    dropout_rate = 0.5
    layer = [nn.Dropout(dropout_rate), nn.Linear(input_size, in_arg.hidden_units[0]), nn.ReLU()]
    for i in range(len(in_arg.hidden_units)):
        layer.append(nn.Dropout(dropout_rate))
        if i >= len(in_arg.hidden_units) -1:
            layer.append(nn.Linear(in_arg.hidden_units[i], output_size))
            layer.append(nn.LogSoftmax(dim=1))
        else:
            layer.append(nn.Linear(in_arg.hidden_units[i], in_arg.hidden_units[i+1]))
            layer.append(nn.ReLU())
    classifier = nn.Sequential(*layer)
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    device = 'cuda' if in_arg.gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Start training on GPU...")
    else:
        print("Start training on CPU...")
        
    
    epochs = in_arg.epochs

    model.to(device)
    for e in range(epochs):
        running_loss = 0
        for images, labels in dataloader:
            #print(labels)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            ps = torch.exp(logps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            model.eval()
            valid_accuracy = 0
            valid_loss = 0
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    optimizer.zero_grad()
                    logps = model.forward(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            print(f"Epoch {e+1}/{epochs}.. ",
                  "Training Loss: {:.3f}.. ".format(running_loss/len(dataloader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(valid_accuracy/len(validloader)))

            model.train()
            
    print("Training finished...")
    print("Saving the model...")
    checkpoint={
        'arch': in_arg.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': train_datasets.class_to_idx,
        'input_size': input_size,
        'output_size': output_size,
        'learning_rate': in_arg.learning_rate,
        'epochs': epochs
    }
    torch.save(checkpoint, in_arg.save_dir + '/checkpoint.pth')
if __name__ == "__main__":
    main()