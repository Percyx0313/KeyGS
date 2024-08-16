# SCENE_LIST=( "Family"  "Ignatius" "Horse"     "Ballroom" "Barn"  "Francis"   "Museum"   "Church" )
SCENE_LIST=( "hydrant" "apple"   "skateboard" "teddybear" "bench" )
for SCENE in "${SCENE_LIST[@]}"
do
ns-render interpolate  --load-config outputs/CO3D_KEY5_BEST/key_gaussian/${SCENE}/config.yml 
mv renders/output.mp4 outputs/CO3D_KEY5_BEST/key_gaussian/${SCENE}/output.mp4
done