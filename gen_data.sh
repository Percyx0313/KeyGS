SCENE_LIST=( "Horse") # "Museum" "Francis"  "Church" "Family"   "Ignatius"  "Ballroom" "Barn" )

for SCENE in "${SCENE_LIST[@]}"
do
    python convert.py -s dataset/Tanks/${SCENE} --sequential
done