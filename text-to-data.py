import os
import shutil
import argparse


def main(args):
    vid_length = args.vid_length
    train_txt_path = args.train_txt_path
    test_txt_path = args.test_txt_path

    # Arrays to contain video names and labels from text files for training and testing sets
    train_vids_names = []
    train_vids_labels = []

    test_vids_names = []
    test_vids_labels = []

    # Open and read the train.txt file, saving the video name as an array
    with open(train_txt_path, 'r') as train_file:
        for line in train_file:
            temp = line.strip()
            name, label = temp.split(' ')
            train_vids_names.append(name)
            train_vids_labels.append(label)

    # Open and read the test.txt file, saving the video name as an array
    with open(test_txt_path, 'r') as test_file:
        for line in test_file:
            temp = line.strip()
            name, label = temp.split(' ')
            test_vids_names.append(name)
            test_vids_labels.append(label)

    # Path to all the videos and path to save the organized dataset for later use
    vid_path = args.vid_path
    org_vid_path = os.path.join(args.org_vid_path + f'_{vid_length}')

    all_vid_folders = os.listdir(vid_path)

    if not os.path.exists(org_vid_path):
        os.mkdir(org_vid_path)

    train_folder = os.path.join(org_vid_path, 'train')
    test_folder = os.path.join(org_vid_path, 'test')

    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    train_vid_paths, test_vid_paths = [], []

    # For each video listed in the train text file, find its path from the videos directory
    for i, vid_name in enumerate(train_vids_names):
        concat_vid_name = None
        if int(train_vids_labels[i]) == 0: # If label is Help, get the full name of the video from the line of the text file
            concat_vid_name = vid_name[:12]
        elif int(train_vids_labels[i]) == 1: # If label is Hinder
            concat_vid_name = vid_name[:14]
        elif int(train_vids_labels[i]) == 2: # If label is Physical
            concat_vid_name = vid_name[:16]
        else: print('The label is not valid or is missing.')

        # print('Video name:', vid_name)
        # print('Folder name:', concat_vid_name)

        # Find the video named in the line of text file in the folder of the dataset videos
        found = False
        for vid in all_vid_folders:
            file_path = os.path.join(vid_path, vid)
            
            if os.path.isfile(file_path) and concat_vid_name in vid:
                train_vid_paths.append(file_path)
                found = True
                break
        if not found: print('The video name is not in the dataset folders.')
        

    # For each video listed in the test file, find its path from the videos directory
    for i, vid_name in enumerate(test_vids_names):
        concat_vid_name = None
        if int(test_vids_labels[i]) == 0:
            concat_vid_name = vid_name[:12]
        elif int(test_vids_labels[i]) == 1:
            concat_vid_name = vid_name[:14]
        elif int(test_vids_labels[i]) == 2:
            concat_vid_name = vid_name[:16]
        else: print('The label is not valid or is missing.')

        # print('Video name:', vid_name)
        # print('Folder name:', concat_vid_name)

        # Find the video named in the line of text file in the folder of the dataset videos
        found = False
        for vid in all_vid_folders:
            file_path = os.path.join(vid_path, vid)
            
            if os.path.isfile(file_path) and concat_vid_name in vid:
                test_vid_paths.append(file_path)
                found = True
                break
        if not found: print('The video name is not in the dataset folders.')


    # Create the data directory for all the train and test videos specified from the text files
    for i, vid_path in enumerate(train_vid_paths):
        label = None
        if int(train_vids_labels[i]) == 0: label = 'Help'
        elif int(train_vids_labels[i]) == 1: label = 'Hinder'
        elif int(train_vids_labels[i]) == 2: label = 'Physical'

        vid_name = os.path.basename(vid_path)

        # print('Video name:', vid_name)

        if not os.path.exists(os.path.join(train_folder, label)):
            os.mkdir(os.path.join(train_folder, label))

        shutil.copy(vid_path, os.path.join(train_folder, label, vid_name))

    for i, vid_path in enumerate(test_vid_paths):
        label = None
        if int(test_vids_labels[i]) == 0: label = 'Help'
        elif int(test_vids_labels[i]) == 1: label = 'Hinder'
        elif int(test_vids_labels[i]) == 2: label = 'Physical'

        vid_name = os.path.basename(vid_path)

        # print('Video name:', vid_name)

        if not os.path.exists(os.path.join(test_folder, label)):
            os.mkdir(os.path.join(test_folder, label))

        shutil.copy(vid_path, os.path.join(test_folder, label, vid_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model")
    parser.add_argument(
        "--vid_length",
        type=int,
        default=100,
        help="Length of video (100%, 67%, 33%)"
    )
    parser.add_argument(
        "--train_txt_path",
        type=str,
        default='./text-files/train_latest.txt',
        help="Path to text file containing list of video names and their classes (i.e., 0=Help, 1=Hinder, 2=Physical) for the training set"
    )
    parser.add_argument(
        "--test_txt_path",
        type=str,
        default='./text-files/test_latest.txt',
        help="Path to text file containing list of video names and their classes (i.e., 0=Help, 1=Hinder, 2=Physical) for the testing set"
    )
    parser.add_argument(
        "--vid_path",
        type=str,
        default='./videos',
        help="Path to all the videos in the custom video dataset"
    )
    parser.add_argument(
        "--org_vid_path",
        type=str,
        default='./org_vid_path',
        help="Output path that will be all the videos in the custom video dataset organized by training and testing sets"
    )
    args = parser.parse_args()
    main(args)