#-#-# Libs:

## Local directory files processing:
from os import listdir # To check files in a folder


## Storage:
from pickle import dump # To store files of model (f. e. features) and dataset in more reliable way (for multiuser using)


## Pre-processing:
import string
import numpy

import torchvision.transforms as transforms # Image transformations
from torch.autograd import Variable
from PIL import Image  # "Load image from file" function


## Models:
from pl_bolts.models.self_supervised import SimCLR # pre-trained SimCLR model (ResNet-50) by Pytorch Ligjhting framework


## PyTorch:
import torch
import torch.nn as nn

#-#-#




#-#-# Functions:

#### Image part functions:

#-# Extract features from each image in the directory:
def extract_features(directory, images_total):

    ## SimCLR model preparation:
    # Initialise pre-trained model with the pre-trained weights for SimCLR model:
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt' 
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False) # SimCLR model initialisation
    simclr.freeze() # Freeze the model parameters for further model using
    print(simclr) # Print the original model

    # Remove the projection head (which needed for training only):
    simclr_resnet50 = simclr.encoder

    # Numerate the model layers (to further remove the last one):
    children_counter = 0
    for n,c in simclr_resnet50.named_children():
        print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1

    # Remove the last fully-connected layer (needed for evaluation on ImageNet only):
    newmodel = torch.nn.Sequential(*(list(simclr_resnet50.children())[:-1])) 
    print(newmodel) # Print the final model

    # Numerate the model layers (to further remove the last one):
    children_counter = 0
    for n,c in newmodel.named_children():
        print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1

    # Set model to the evaluation state:
    newmodel.eval() 


    ## Image data pre-processing subfunctions:
    scaler = transforms.Resize((224, 224)) # Resize input image to 224x224 shape

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                    # Normilise image

    to_tensor = transforms.ToTensor() # Put normilised image in Tensor


        ## Extract features from each image:
    features = dict() # Create a dictionary of "image - features"

    image_number = 0 # Image index (for further printing a ratio of processed images in dataset)

    for name in listdir(directory): # For each image in the dataset
        ## Load an image from file:
        filename = directory + '/' + name

        # Load the image with Pillow library
        img = Image.open(filename)


        ## Image preprocessing:
        # Create a PyTorch variable with the transformed image
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))


        ## Get features from the last model's layer:
        feature_tens = newmodel(t_img) # Get features
        feature_nump = feature_tens.detach().cpu().numpy() # Convert them to the numpy array
        feature = feature_nump[:, :, 0, 0] # Reshape the array

        ## Image post-processing:
        image_id = name.split('.')[0] # Get image ID

        features[image_id] = feature # Store obtained feature

        #print('>%s' % name) Print the name of processed image


        ## Print current % of processed images:
        if(int(image_number % 100) == 0):
            print('Percentage of pre-processed images: ', int((image_number / images_total) * 100), '%')

        image_number = image_number + 1 # Update image index


    ## Return the feature vector:
    return features
#-#



#### Text part functions:

#-# Load text dataset:
def load_text(filename):
    file = open(filename, 'r') # Open the file as read only
    text = file.read() # Read all text
    file.close() # Close the file

    return text
#-#


#-# Extract descriptions for images from the text dataset:
def load_descriptions(file):
    descriptions = dict() # Will be a dictionary "Image_id - [List of Descriptions]":

    i = 0

    for line in file.split('\n'): # Split lines by "new line" character (when user input "Enter" for the next line)
        i = i + 1

        tokens = line.split() # Split lines onto tokens (sub-texts, divided by "1 space")

        if len(line) < 2: # Check if line is empty (if so, skip this line and go to the beginning of the loop)
            # Actually, we need this condition to skip the unreal "last" line after the last "\n" after the real last line
            # Example:
                # "My cool line" \n
            # We can improve this by use of another splitting way 

            print("Empty line is in the row number: ", i)
            continue

        image_id, image_desc = tokens[0], tokens[1:] # 1st token - image id, the rest - description
        image_id = image_id.split('.')[0] # Remove filename from image id
        image_desc = ' '.join(image_desc) # Convert description tokens back to 1 string

        ## Fill up a dictionary "Image_name - Descriptions":
        if image_id not in descriptions: # If image_id (uniq id) is not in a dictionary (Because we have more than 1 description for most of the images)
            descriptions[image_id] = list() # Create an empty list for further descriptions of this image

        descriptions[image_id].append(image_desc) # Add deription to the correspondiong images

    return descriptions
#-#


#-# Clean text of descriptions (To reduce the size of a vocabulary of words we will use)
    # Rules of cleaning:
    # - Convert all words to lowercase.
    # - Remove all punctuation.
    # - Remove all words that are one character or less in length (e.g. ‘a’).
    # - Remove all words with numbers in them.

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation) # Prepare "translation" table for removing punctuation

    for key, desc_list in descriptions.items(): # In the dictionary "Image_name - Descriptions" [mapping]
        for i in range(len(desc_list)): # Check all the descriptions (one-by-one) of an image
            desc = desc_list[i] # Choose a certain description
            desc = desc.split() # Tokenise (Divide word-by-word) a certain description

            ## Apply cleaning of descriptions:
            desc = [word.lower() for word in desc] # Convert to lower case
            desc = [w.translate(table) for w in desc] # Remove punctuation from each token
            desc = [word for word in desc if len(word)>1] # Remove words of 1 character (like "'s" and "a")
            desc = [word for word in desc if word.isalpha()] # Remove tokens with numbers in them

            desc_list[i] =  ' '.join(desc) # Combine a result and store it as string

    # P. S. No need to return anything because we have applied cleaning directly to the dictionary of descriptions
#-#


#-# Save descriptions in a file (one per line):
def save_descriptions(descriptions, filename):
    lines = list() # Create an empty list (separated on the lines by \n ["enter"]) for all descriptions ("image_id + 1 description")
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc) # Create 

    data = '\n'.join(lines) # Convert a list of lines in a string text, and add a separation on lines by \n ["enter"]

    ## Write the result in a file:
    file = open(filename, 'w') # Open file with writing permission
    file.write(data) # Write text in file
    file.close() # Close file
#-#


#-# Convert the loaded dictionary "Image_name - Descriptions" of cleaned descriptions into a vocabulary of used words (the less the better)
def to_vocabulary(descriptions):
    all_desc = set() # Create an empty list for all used words

    ## Fill up the list with words from descriptions:
    for key in descriptions.keys(): # Use already created dictionary "Image_name - Descriptions" of descriptions
        [all_desc.update(d.split()) for d in descriptions[key]] # For every image split its every description into words and add them into the list of used words

    return all_desc
#-#

#-#-#




#-#-# Main:

if __name__ == '__main__': # !Don't forget to change the dataset paths!

    ### Preparation code:
    ## Images part:
    directory = 'Flicker8k_Dataset' # Dataset of images directory (folder name)
    #directory = '/home/dd/Documents/Datasets/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/'

    # To calculate percantage of pre-processed images:
    img_folder_path = directory
    dirListing = listdir(img_folder_path) # Get list of files in Dataset directory
    images_total = len(dirListing) # Number of files in the Dataset folder


    ## Text part:
    filename = 'Flickr8k_text/Flickr8k.token.txt' # Dataset of text directory (folder name + file name) - txt. file
    #filename = '/home/dd/Documents/Datasets/Flickr8k/Flickr8k_text/Flickr8k.token.txt'

    text = load_text(filename) # Load text dataset
    descriptions = load_descriptions(text) # Load descriptions from the file to a dictionary "Image_name - Descriptions"
    print('Loaded number of descriptions: %d ' % len(descriptions))



    ### Main code:
    ## Images part:
    # Extract features from all images:
    features = extract_features(directory, images_total)
    print('Number of Extracted from Images Features: %d' %( len(features)) )

    # Save features to file:
    dump(features, open('features.pkl', 'wb'))


    ## Text part:
    # Pre-process descriptions:
    clean_descriptions(descriptions) # Clean descriptions (Remove punctuation, etc...)
    save_descriptions(descriptions, 'descriptions.txt') # Save vocabulary to file

    # Create a vocabulary of used words in descriptions:
    vocabulary = to_vocabulary(descriptions) #  Create a vocabulary vocabulary of used words (from the dictionary of cleaned escriptors "Image_name - Descriptions")
    print('Number of Words in Text Vocabulary: %d' % len(vocabulary))

#-#-#