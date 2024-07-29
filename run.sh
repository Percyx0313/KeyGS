export NERFSTUDIO_METHOD_CONFIGS="key_gaussian=key_gaussian.key_gaussian_config:key_gaussian"

# ns-train splatfacto --data data/Museum_Key5/ \
#     --optimizers.camera-opt.optimizer.lr=1e-3 \
#     --viewer.quit_on_train_completion True \
#     --vis viewer+tensorboard \
#     colmap --colmap-path sparse/0 --auto-scale-poses False

# SCENE_LIST=( "Horse" "Museum"   "Church" "Family"   "Ignatius"  "Ballroom" "Barn" "Francis" )
SCENE_LIST=(   "Francis")
EXP_NAME="base+abs+cull+antialiased+split25k+inter"
for SCENE in "${SCENE_LIST[@]}"
do
mkdir -p outputs/${SCENE}/key_gaussian/${EXP_NAME}/
# training  # current best
# ns-train key_gaussian --data Tanks_key5/${SCENE} \
#     --optimizers.camera-opt.optimizer.lr 1e-3 \
#     --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
#     --vis tensorboard \
#     --timestamp ${EXP_NAME} \
#     --pipeline.model.stop-split-at 15000 \
#     --pipeline.model.sh-degree-interval 1000 \
#     --pipeline.model.rasterize-mode antialiased \
#     --pipeline.model.camera-optimizer.mode SO3xR3 \
#     --optimizers.camera-opt.scheduler.warmup-steps 3000 \
#     --pipeline.model.cull-alpha-thresh 0.05 \
#     --pipeline.model.use-scale-regularization True \
#     colmap --colmap-path sparse/0 --auto-scale-poses True \
#     | tee -a outputs/${SCENE}/${EXP_NAME}/train_log.txt 


ns-train key_gaussian --data dataset/Tanks_inter/${SCENE} \
    --optimizers.camera-opt.optimizer.lr 1e-3 \
    --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
    --vis tensorboard \
    --timestamp ${EXP_NAME} \
    --pipeline.model.stop-split-at 20000 \
    --pipeline.model.sh-degree-interval 1000 \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.camera-optimizer.mode SO3xR3 \
    --optimizers.camera-opt.scheduler.warmup-steps 1000 \
    --pipeline.model.cull-alpha-thresh 0.05 \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.use-abs-grad True \
    --pipeline.model.densify-grad-thresh 0.0004 \
    --pipeline.model.split-screen-size 0.05 \
    --pipeline.model.stop-screen-size-at 4000 \
    colmap --colmap-path sparse/0 --auto-scale-poses True \
    | tee -a outputs/${SCENE}/key_gaussian/${EXP_NAME}/train_log.txt 
# export camera pose
ns-export cameras --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-dir outputs/${SCENE}/key_gaussian/${EXP_NAME}/ | tee -a outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval_log.txt 
ns-eval --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-path outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval.json --render-output-path outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval/ 
done

# move to the final directory 
# mkdir -p outputs/${EXP_NAME}
# for SCENE in "${SCENE_LIST[@]}"
# do
#     mv outputs/${SCENE}/key_gaussian/${EXP_NAME} outputs/${EXP_NAME}/${SCENE}
#     rm -r outputs/${SCENE}
# done

