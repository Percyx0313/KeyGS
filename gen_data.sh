# SCENE_LIST=(  "Church" "Ignatius" "Horse" "Museum" "Francis"   "Family"   "Ignatius"  "Ballroom" "Barn" )
SCENE_LIST=( "Francis" ) #"Barn" "Ballroom" )

for SCENE in "${SCENE_LIST[@]}"
do
    # mv dataset/Tanks_inter/${SCENE}/images dataset/Tanks_inter/${SCENE}/raw_images
    python convert.py -s dataset/Tanks_inter/${SCENE} --sequential
    # hand convert 
    colmap model_converter --input_path dataset/Tanks/${SCENE}/sparse/0 --output_path dataset/Tanks/${SCENE}/sparse/0 --output_type TXT
done