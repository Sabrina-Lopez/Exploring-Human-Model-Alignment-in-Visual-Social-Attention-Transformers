import os
import cv2
import datetime
import csv
import argparse


def main(args):
  dataset = args.org_vid_path
  train_data_csv_name = os.path.basename(dataset) + '_' + args.train_csv
  test_data_csv_name = os.path.basename(dataset) + '_' + args.test_csv

  # Define output CSV files for train, test, and validation
  csv_outputs = {
      'train': train_data_csv_name,
      'test': test_data_csv_name
  }

  # Create CSV files and write headers
  for data_type, csv_output in csv_outputs.items():
    with open(csv_output, 'w', newline='') as file:
      writer = csv.writer(file)
      field = ['video_path', 'start', 'end', 'label']
      writer.writerow(field)

      cur_data = os.path.join(dataset, data_type)
      
      for cur_class in os.listdir(cur_data):  # This is the label for all the videos in the third for-loop
        cur_video_class = os.path.join(dataset, data_type, cur_class)
        
        for video in os.listdir(cur_video_class):
          cur_video = os.path.join(dataset, data_type, cur_class, video)
          
          # Check if the file exists
          if not os.path.exists(cur_video):
              print('File not found: {}'.format(cur_video))
              continue
          
          cap_video = cv2.VideoCapture(cur_video)
          
          frames = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
          fps = cap_video.get(cv2.CAP_PROP_FPS)
          
          seconds = round(frames / fps) # Convert to total seconds
          video_time = datetime.timedelta(seconds=seconds)
          seconds_as_float = float(video_time.total_seconds())
          
          start_time = 0.0

          label = None
          if cur_class == 'Help': label = 0
          elif cur_class == 'Hinder': label = 1
          elif cur_class == 'Physical': label = 2
          
          path = '{}/{}/{}/{}'.format(dataset, data_type, cur_class, video)
          writer.writerow([f'{path}', f'{start_time}', f'{seconds_as_float}', f'{label}'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model")
    parser.add_argument(
        "--org_vid_path",
        type=str,
        default='./org_vid_path_100',
        help="Path to all the videos in the custom video dataset organized by training and testing sets"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default='train.csv',
        help="Path to all the videos in the custom video dataset organized by training and testing sets"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default='test.csv',
        help="Path to all the videos in the custom video dataset organized by training and testing sets"
    )
    args = parser.parse_args()
    main(args)