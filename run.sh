export NERFSTUDIO_METHOD_CONFIGS="key_gaussian=key_gaussian.key_gaussian_config:key_gaussian"

# ns-train splatfacto --data data/Museum_Key5/ \
#     --optimizers.camera-opt.optimizer.lr=1e-3 \
#     --viewer.quit_on_train_completion True \
#     --vis viewer+tensorboard \
#     colmap --colmap-path sparse/0 --auto-scale-poses False

# SCENE_LIST=( "Horse" "Museum" "Francis"  "Church" "Family"   "Ignatius"  "Ballroom" "Barn" )
# EXP_NAME="Key5"
# for SCENE in "${SCENE_LIST[@]}"
# do
# mkdir -p Tanks_key5/${SCENE}
# ns-train key_gaussian --data Tanks_key5/${SCENE} \
#     --pipeline.model.cull-screen-size 0.5 \
#     --optimizers.camera-opt.optimizer.lr 1e-3 \
#     --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
#     --vis tensorboard \
#     --timestamp ${EXP_NAME}\
#     colmap --colmap-path sparse/0 --auto-scale-poses True | tee -a outputs/${SCENE}/${EXP_NAME}/train_log.txt 

# done

SCENE="Horse"
EXP_NAME="With_interpolation_pose"
mkdir -p Tanks/${SCENE}
mkdir -p outputs/${SCENE}/${EXP_NAME}
# # training
# ns-train key_gaussian --data Tanks/${SCENE} \
#     --pipeline.model.cull-screen-size 0.5 \
#     --optimizers.camera-opt.optimizer.lr 1e-3 \
#     --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
#     --vis tensorboard \
#     --timestamp ${EXP_NAME}\
#     colmap --colmap-path sparse/0 --auto-scale-poses True | tee -a outputs/${SCENE}/${EXP_NAME}/train_log.txt 

# training 
ns-train key_gaussian --data Tanks/${SCENE} \
    --optimizers.camera-opt.optimizer.lr 1e-3 \
    --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
    --vis tensorboard \
    --timestamp ${EXP_NAME}\
    --pipeline.model.stop-split-at 15000 \
    --pipeline.model.sh-degree-interval 1000 \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.camera-optimizer.mode SO3xR3 \
    --optimizers.camera-opt.scheduler.warmup-steps 3000 \
    --pipeline.model.cull-alpha-thresh 0.05 \
    --pipeline.model.use-scale-regularization True \
    colmap --colmap-path sparse/0 --auto-scale-poses True \
    | tee -a outputs/${SCENE}/${EXP_NAME}/train_log.txt 
# export camera pose

ns-export cameras --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-dir outputs/${SCENE}/key_gaussian/${EXP_NAME}/