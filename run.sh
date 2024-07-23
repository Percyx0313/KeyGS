export NERFSTUDIO_METHOD_CONFIGS="key_gaussian=key_gaussian.key_gaussian_config:key_gaussian"

# ns-train splatfacto --data data/Museum_Key5/ \
#     --optimizers.camera-opt.optimizer.lr=1e-3 \
#     --viewer.quit_on_train_completion True \
#     --vis viewer+tensorboard \
#     colmap --colmap-path sparse/0 --auto-scale-poses False

SCENE_LIST=( "Horse" "Museum" "Francis"  "Church" "Family"   "Ignatius"  "Ballroom" "Barn" )
EXP_NAME="Key5"
for SCENE in "${SCENE_LIST[@]}"
do
mkdir -p Tanks_key5/${SCENE}
ns-train key_gaussian --data Tanks_key5/${SCENE} \
    --pipeline.model.cull-screen-size 0.5 \
    --optimizers.camera-opt.optimizer.lr 1e-3 \
    --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
    --vis tensorboard \
    --timestamp ${EXP_NAME}\
    colmap --colmap-path sparse/0 --auto-scale-poses True | tee -a outputs/${SCENE}/${EXP_NAME}/train_log.txt 

done