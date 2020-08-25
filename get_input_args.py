import argparse

def get_training_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', type = str, default = 'flowers')
    
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', 
                    help = 'path to the folder to save the checkpoint. Default is checkpoints') 
    
    parser.add_argument('--arch', type = str, default = 'vgg19', 
                    help = 'CNN model architecture. Default is vgg19') 
    
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'Learning_rate. Default is 0.1') 
    
    parser.add_argument('--hidden_units', type = int, default = [1024], nargs='*',
                    help = 'Hidden units. Default is 1024') 
    
    parser.add_argument('--epochs', type = int, default = 10,
                    help = 'Epochs. Default is 10') 
    
    parser.add_argument('--gpu', action='store_true', help = 'Use GPU for training')
    
    
    return parser.parse_args() 


def get_prediction_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type = str, default = 'flowers/test/10/image_07090.jpg')
    
    parser.add_argument('checkpoint_directory', type = str, default = 'checkpoints')
    
    parser.add_argument('--top_k', type = int, default = 5, 
                    help = 'Number of top classes to return. Default is 5') 
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'Path to category name json file. Default is cat_to_name.json') 
    
    parser.add_argument('--gpu', action='store_true', help = 'Use GPU for training')
    
    
    return parser.parse_args() 