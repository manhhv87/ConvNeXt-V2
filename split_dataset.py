import argparse
import os
import random
import shutil

datasetFolderName = '/home/hanh.buithi/pytorch/ConvNeXt-V2/dataset'
sourceFiles = []
classLabels = sorted(os.listdir('/home/hanh.buithi/pytorch/ConvNeXt-V2/dataset/train'))


def get_args_parser():
    parser = argparse.ArgumentParser('Create dataset', add_help=False)
    parser.add_argument('--train_path', default='train', type=str,
                        help='Train set path')
    parser.add_argument('--val_path', default='val', type=str,
                        help='Validation set path')
    parser.add_argument('--test_path', default='test', type=str,
                        help='Test set path')
    return parser
    

def transferBetweenFolders(source, dest, splitRate):
    global sourceFiles
    sourceFiles = os.listdir(source)
    if(len(sourceFiles) != 0):
        transferFileNumbers = int(len(sourceFiles)*splitRate)
        transferIndex = random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source + str(sourceFiles[eachIndex]), dest + str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName + '/' + source + '/' + label + '/',
                               datasetFolderName + '/' + dest + '/' + label + '/',
                               splitRate)
                               

def main(args):
    transferAllClassBetweenFolders('train', 'val', 0.4)
    transferAllClassBetweenFolders('val', 'test', 0.5)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
                                   
                          