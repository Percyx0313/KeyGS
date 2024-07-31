# # SCENE_LIST=(  "Church" "Ignatius" "Horse" "Museum" "Francis"   "Family"   "Ignatius"  "Ballroom" "Barn" )

# for SCENE in "${SCENE_LIST[@]}"
# do
#     mv dataset/Tanks_inter/${SCENE}/images dataset/Tanks_inter/${SCENE}/raw_images
#     python convert.py -s dataset/Tanks_inter/${SCENE} --sequential
#     # hand convert 
#     colmap model_converter --input_path dataset/Tanks/${SCENE}/sparse/0 --output_path dataset/Tanks/${SCENE}/sparse/0 --output_type TXT
# done

SCENE_LIST=(  "apple/110_13051_23361" "bench/415_57112_110099"  "skateboard/245_26182_52130" "hydrant/106_12648_23157" "teddybear/34_1403_4393")
# SCENE_LIST=(  "hydrant/106_12648_23157" "teddybear/34_1403_4393")
SCENE_LIST=(   "teddybear/34_1403_4393")

for SCENE in "${SCENE_LIST[@]}"
do
    # mv dataset/CFGS_Co3d_inter/${SCENE}/images dataset/CFGS_Co3d_inter/${SCENE}/raw_images
    python -u convert.py -s dataset/CFGS_Co3d_inter/${SCENE} --sequential  --keyframe_interval 5 --overlap 20 --skip_matching | tee -a dataset/CFGS_Co3d_inter/${SCENE}/log.txt 
    # hand convert 
    colmap model_converter --input_path dataset/CFGS_Co3d_inter/${SCENE}/sparse/0 --output_path dataset/CFGS_Co3d_inter/${SCENE}/sparse/0 --output_type TXT
done