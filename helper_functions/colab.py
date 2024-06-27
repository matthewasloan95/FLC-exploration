import copy
import os
from google.colab import drive
from ultralytics.data.converter import convert_coco
import glob
import random
import locale
from pathlib import Path

def install_packages():
    # !pip install ultralytics
    # !pip install comet_ml
    
    # os pip install
    os.system('pip install -U comet_ml ultralytics')
    
def mount_drive():
    drive.mount('/content/drive')
    
def unzip_file(zip_file_path='/content/drive/MyDrive/clovers/1280_flc.zip', destination_path='/content/flc/'):
    os.system(f'!7z x {zip_file_path} -mmt=on -o{destination_path}')
    
def copy_file(source_path='/content/drive/MyDrive/clovers/FLCv3.yaml', destination_path='/content'):
    os.system(f'!cp {source_path} {destination_path}')

def convert_coco_annotations(annotations_path=['/content/flc/1280_flc/trainval/coco_annotations', '/content/flc/1280_flc/test/coco_annotations'], use_segments=True, use_keypoints=False, cls91to80=True, copy_from=['/content/coco_converted/labels/trainval_pos/*.txt', '/content/coco_converted2/labels/test_pos/*.txt'], copy_to=['/content/flc/1280_flc/trainval/JPEGImages_1280x720', '/content/flc/1280_flc/test/JPEGImages_1280x720']):
    
    for path in annotations_path:
        convert_coco(path, use_segments=use_segments, use_keypoints=use_keypoints, cls91to80=cls91to80)
    
    for copy_fr, copy_t in zip(copy_from, copy_to):
        os.system(f'!cp {copy_fr} {copy_t}')
    
    
def misc_rm():
    os.system('rm /content/flc/1280_flc/test/coco_annotations/leaves_test_negs.json')
    
    os.system('rm /content/flc/1280_flc/test/coco_annotations/leaves_test_pos.json')
    
    os.system('rm /content/flc/1280_flc/test/coco_annotations/instances_test_hard_pos.json')
    
    os.system('rm /content/flc/1280_flc/test/coco_annotations/instances_test_negs.json')
    
    os.system('rm /content/flc/1280_flc/test/coco_annotations/instances_test_all.json')
    
    os.system('rm /content/flc/1280_flc/trainval/coco_annotations/instances_trainval_hard_pos.json')

    os.system('rm /content/flc/1280_flc/trainval/coco_annotations/instances_trainval_negs.json')
    
    os.system('rm /content/flc/1280_flc/trainval/coco_annotations/leaves_trainval_negs.json')

    os.system('rm /content/flc/1280_flc/trainval/coco_annotations/leaves_trainval_pos.json')
    
def check_files(directory):
    missing_or_empty_files = []

    # Check all .jpg files that match the pattern 1*.jpg
    for jpg_file in glob.glob(os.path.join(directory, '1*.jpg')):
        base_name = os.path.splitext(jpg_file)[0]
        txt_file = base_name + '.txt'

        # Check if the corresponding .txt file exists
        if not os.path.isfile(txt_file):
            missing_or_empty_files.append(jpg_file)
        else:
            # Check if the .txt file is empty
            if os.path.getsize(txt_file) == 0:
                missing_or_empty_files.append(txt_file)

    return missing_or_empty_files

def adjust_dataset_balance(directory, percent_negatives):
    """
    Adjusts the dataset to have the specified percentage of negative images relative to positive images.

    Parameters:
    - directory: Path to the directory containing the images.
    - percent_negatives: The target percentage of negative images in the dataset.

    The function ensures that the number of negative images is adjusted to match the specified percentage
    relative to the number of positive images.
    """
    # Count positive images
    positive_images = glob.glob(os.path.join(directory, "1*.jpg"))
    num_positives = len(positive_images)

    # Calculate the target number of negatives based on the desired percentage
    target_num_negatives = int((num_positives * percent_negatives) / (100 - percent_negatives))

    # Step 1: Rename .jpg.bak to .jpg for all negative images to make them available for counting and adjustment
    for filepath in glob.glob(os.path.join(directory, "0*.jpg.bak")):
        new_path = filepath[:-4]  # Remove '.bak'
        os.rename(filepath, new_path)
        # print(f"Renamed {filepath} to {new_path}")

    # Get current list of negative .jpg images
    negative_images = glob.glob(os.path.join(directory, "0*.jpg"))
    current_num_negatives = len(negative_images)

    # Determine adjustment needed based on current number of negatives
    if current_num_negatives > target_num_negatives:
        # Too many negatives, select some to rename to .bak
        num_to_rename = current_num_negatives - target_num_negatives
        images_to_rename = random.sample(negative_images, num_to_rename)
        for filepath in images_to_rename:
            new_path = filepath + ".bak"
            os.rename(filepath, new_path)
            # print(f"Renamed {filepath} to {new_path}")
        print('finished if')

    elif current_num_negatives < target_num_negatives:
        # Not enough negatives, try to rename .bak files back to .jpg if available
        additional_needed = target_num_negatives - current_num_negatives
        backup_negatives = glob.glob(os.path.join(directory, "0*.jpg.bak"))
        if additional_needed <= len(backup_negatives):
            images_to_rename = random.sample(backup_negatives, additional_needed)
            for filepath in images_to_rename:
                new_path = filepath[:-4]  # Remove '.bak'
                os.rename(filepath, new_path)
                # print(f"Renamed {filepath} to {new_path}")
            print('finished elif if')
        else:
            print("Warning: Not enough negative .bak files to reach the desired ratio.")
        print('finished elif else')
    print('ALL finished')

def next_run(percent_neg: int, rm_runs=True):
  locale.getpreferredencoding = lambda: "UTF-8"
  if rm_runs:

    os.system('rm -rf /content/runs/')

    os.system('rm /content/flc/1280_flc/trainval/labels.cache')

    os.system('rm /content/flc/1280_flc/test/labels.cache')

  directory_path = "/content/flc/1280_flc/test/images"
  # percent_negatives = 50  # To match the number of negatives to 50% of the positives
  adjust_dataset_balance(directory_path, percent_neg)

  directory_path = "/content/flc/1280_flc/trainval/images"
  # percent_negatives = 50  # To match the number of negatives to 50% of the positives
  adjust_dataset_balance(directory_path, percent_neg)



def create_empty_txt_files(source_dir, target_dir):
    # Ensure target directory exists
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Pattern to match "0*.jpg" files
    pattern = os.path.join(source_dir, '0*.jpg')

    for jpg_file in glob.glob(pattern):
        # Extract base name and create corresponding .txt file name
        base_name = os.path.basename(jpg_file)
        txt_file_name = os.path.splitext(base_name)[0] + '.txt'

        # Paths for the new .txt files in both directories
        source_txt_path = os.path.join(source_dir, txt_file_name)
        target_txt_path = os.path.join(target_dir, txt_file_name)

        # Create an empty .txt file in the source directory
        open(source_txt_path, 'w').close()

        # Create an empty .txt file in the target directory
        open(target_txt_path, 'w').close()
        
def prep():
    install_packages()
    mount_drive()
    unzip_file()
    copy_file()
    convert_coco_annotations()
    misc_rm()
    print('----- TEST DIR -----')
    result = check_files('/content/flc/1280_flc/test/JPEGImages_1280x720')
    print(result)
    print('----- TRAIN DIR -----')
    result = check_files('/content/flc/1280_flc/trainval/JPEGImages_1280x720')
    print(result)
    
    
    # !rm /content/flc/1280_flc/test/JPEGImages_1280x720/1_000301.jpg
    os.system('rm /content/flc/1280_flc/test/JPEGImages_1280x720/1_000301.jpg')
    # !rm /content/flc/1280_flc/trainval/JPEGImages_1280x720/1_000983.jpg
    os.system('rm /content/flc/1280_flc/trainval/JPEGImages_1280x720/1_000983.jpg')

    # !mkdir /content/flc/1280_flc/test/labels/
    os.system('mkdir /content/flc/1280_flc/test/labels/')
    # !mkdir /content/flc/1280_flc/trainval/labels/
    os.system('mkdir /content/flc/1280_flc/trainval/labels/')
    # !cp /content/coco_converted/labels/trainval_pos/*.txt /content/flc/1280_flc/trainval/labels/
    os.system('cp /content/coco_converted/labels/trainval_pos/*.txt /content/flc/1280_flc/trainval/labels/')
    # !cp /content/coco_converted2/labels/test_pos/*.txt /content/flc/1280_flc/test/labels/
    os.system('cp /content/coco_converted2/labels/test_pos/*.txt /content/flc/1280_flc/test/labels/')

    # !rm -rf /content/coco_converted
    os.system('rm -rf /content/coco_converted')
    # !rm -rf /content/coco_converted2  
    os.system('rm -rf /content/coco_converted2')
    
    # !rm -rf /content/coco_converted3
    os.system('rm -rf /content/coco_converted3')
    # !rm -rf /content/coco_converted4
    os.system('rm -rf /content/coco_converted4')
    # !rm -rf /content/coco_converted5
    os.system('rm -rf /content/coco_converted5')
    # !rm -rf /content/coco_converted6
    os.system('rm -rf /content/coco_converted6')

    # !mv /content/flc/1280_flc/test/JPEGImages_1280x720 /content/flc/1280_flc/test/images
    os.system('mv /content/flc/1280_flc/test/JPEGImages_1280x720 /content/flc/1280_flc/test/images')
    # !mv /content/flc/1280_flc/trainval/JPEGImages_1280x720 /content/flc/1280_flc/trainval/images
    os.system('mv /content/flc/1280_flc/trainval/JPEGImages_1280x720 /content/flc/1280_flc/trainval/images')
    
    create_empty_txt_files('/content/flc/1280_flc/test/images', '/content/flc/1280_flc/test/labels')
    create_empty_txt_files('/content/flc/1280_flc/trainval/images', '/content/flc/trainval/1280_flc/labels')
    
    next_run(percent_neg=50)