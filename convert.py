#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import glob
import time
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--sequential", action='store_true')
parser.add_argument("--overlap", default=5, type=int)
parser.add_argument("--keyframe_interval", default=5, type=int)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0
start_time=time.time()

if not args.skip_matching:
    ## extract the key frame to input folder
    raw_images_file_list=glob.glob(args.source_path+"/raw_images/*.jpg")
    if raw_images_file_list==[]:
        raw_images_file_list=glob.glob(args.source_path+"/raw_images/*.png")
    assert raw_images_file_list!=[] , "There are no images in the folder"
    raw_images_file_list=sorted(raw_images_file_list)
    
    input_path=args.source_path+'/input'
    os.makedirs(input_path,exist_ok=True)
    
    # extract the key frame by the keyframe_interval
    key_frame_image_file_list=raw_images_file_list[::args.keyframe_interval]
    
    # add the last frame for convinience to interpolation
    if(raw_images_file_list[-1] not in key_frame_image_file_list):
        key_frame_image_file_list.append(raw_images_file_list[-1])
    
    
    for fh in key_frame_image_file_list:
        shutil.copy2(fh,input_path)
    
    
    
    
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) 
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print(f"using overlap {args.overlap}")
    SUCCESS=False
    # while not SUCCESS:
    ## Feature matching
    if args.sequential==True:
        feat_matching_cmd = colmap_command + " sequential_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu) + " --SequentialMatching.overlap " + str(args.overlap) + " --SequentialMatching.loop_detection_num_nearest_neighbor 10"
    else:
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
        # if os.path.exists(args.source_path + "/distorted/sparse/1"):
        #     args.overlap+=5
        #     print(f"using overlap {args.overlap}")
        #     fail_dir=glob.glob(args.source_path + "/distorted/sparse/*")
        #     for fh in fail_dir:
        #         shutil.rmtree(fh)
        #     if args.overlap==25:
        #         logging.error(f"Mapper failed.")
        #         break
        # else:
        #     SUCCESS=True

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)
    
# convert the .bin to .txt
convert_bin2txt_cmd=feat_matching_cmd = colmap_command + " model_converter \
            --input_path " + args.source_path + "/sparse/0/ \
            --output_path " +  args.source_path + "/sparse/0/" + " --output_type TXT"

exit_code = os.system(feat_matching_cmd)
if exit_code != 0:
    logging.error(f"model_converter failed with code {exit_code}. Exiting.")
    exit(exit_code)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)



end_time=time.time()
print("Total cost time {} (s)".format(end_time-start_time))
print("Done.")