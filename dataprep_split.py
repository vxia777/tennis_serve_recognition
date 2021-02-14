import csv
import os, sys
import os.path
import random

def train_dev_test_split(train_pctg=0.8,
                         dev_pctg=0.1,
                         filesavename='data/data_file.csv'):
    """
    Split the data into training/dev/test through a train_pctg/dev_pctg/(1 - train_pctg - dev_pctg) split.
    The split is done independently within each class so that the class distribution is maintained.
    This function requires that all of the raw .avi video data be stored in a 'VIDEO_RGB' subfolder prior to running.
    It saves a csv file into the filesavename location (must have the data folder be created before running).
    :param train_pctg: % of data to use for training (e.g. 80%)
    :param dev_pctg: % of data to use for dev (e.g. 10%)... remainder of data after train/dev to be used for test
    :param filesavepath: file path and name to save the file -- must be within data subfolder
    :return:
    """

    with open(filesavename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # get subdirectories in VIDEO_RGB
        video_RGB_path = os.path.join(os.getcwd(), 'VIDEO_RGB')

        # only grab service folders (ignoring .DS_store type files as well)
        serve_folders = [x for x in os.listdir(video_RGB_path) if 'service' in x]

        # iterate over each folder, and write train/dev/test samples to csv file
        for serve_label in serve_folders:
            subpath = os.path.join(video_RGB_path, serve_label)

            videos = []
            for vid in os.listdir(subpath):

                # make sure vid is an actual file and not .DS_Store type
                if os.path.isfile(os.path.join(subpath, vid)) and not vid.startswith('.'):
                    vid = os.path.splitext(vid)[0]  # get the filename before .avi
                    videos.append(vid)

            # shuffle the video names in place
            random.shuffle(videos)

            # split into training/dev/test sets
            train_split_ind = int(round(train_pctg * len(videos)))
            dev_split_ind = train_split_ind + int(round(dev_pctg * len(videos)))

            train = videos[0:train_split_ind]
            dev = videos[train_split_ind:dev_split_ind]
            test = videos[dev_split_ind:]

            # write output to a 3-column csv file:
            # column 1 -- whether the file is used for train, dev, or test
            # column 2 -- name of the class (the type of serve)
            # column 3 -- name of the file (excluding .avi)
            for sample in train:
                writer.writerow(['train'] + [serve_label] + [sample])

            # write val_size proportion to csv as validation
            for sample in dev:
                writer.writerow(['dev'] + [serve_label] + [sample])

            # write remaining proportion to csv as test
            for sample in test:
                writer.writerow(['test'] + [serve_label] + [sample])

    return

if __name__ == "__main__":
    # set random seed
    random.seed(1)

    train_dev_test_split()