export NERFSTUDIO_METHOD_CONFIGS="key_gaussian=key_gaussian.key_gaussian_config:key_gaussian"
# Baseline model
# SCENE_LIST=(  "Family"  "Ignatius" "Horse"     "Ballroom" "Barn"  "Francis"   "Museum"   "Church" )

# EXP_NAME="TANK_KEY10_EXHAUSTIVE" 
# for SCENE in "${SCENE_LIST[@]}"
# do
# mkdir -p outputs/${SCENE}/key_gaussian/${EXP_NAME}/
# ns-train key_gaussian --data dataset/Tanks/${SCENE} \
#     --optimizers.camera-opt.optimizer.lr 1e-3 \
#     --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
#     --vis tensorboard \
#     --timestamp ${EXP_NAME} \
#     --pipeline.model.stop-split-at 20000 \
#     --pipeline.model.sh-degree-interval 1000 \
#     --pipeline.model.rasterize-mode antialiased   --pipeline.model.camera-optimizer.mode SO3xR3 \
#     --optimizers.camera-opt.scheduler.warmup-steps 1000 \
#     --pipeline.model.output-depth-during-training True \
#     --pipeline.model.cull-alpha-thresh 0.05 \
#     --pipeline.model.densify-grad-thresh 0.0004 \
#     --optimizers.camera-opt.scheduler.lr-pre-warmup 1e-5 --pipeline.model.use-abs-grad True \
#     --pipeline.model.background-color black \
#     --pipeline.model.split-screen-size 0.05 \
#     --pipeline.model.stop-screen-size-at 4000 \
#     --pipeline.model.camera-optimizer.trans-l2-penalty 0.0 \
#     --pipeline.model.camera-optimizer.rot-l2-penalty  0.0 \
#     --pipeline.model.use-scale-regularization True \
#     colmap --colmap-path sparse/0 --auto-scale-poses True \
#     | tee -a outputs/${SCENE}/key_gaussian/${EXP_NAME}/train_log.txt 
# # export camera pose
#     mv train_temp outputs/${SCENE}/key_gaussian/${EXP_NAME}/
#     ns-export cameras --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-dir outputs/${SCENE}/key_gaussian/${EXP_NAME}/ 
#     ns-eval --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-path outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval.json --render-output-path outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval/  | tee -a outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval_log.txt 
#     ns-render interpolate  --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml
#     mv renders/output.mp4 outputs/${SCENE}/key_gaussian/${EXP_NAME}/output.mp4
# done

# mkdir -p outputs/${EXP_NAME}
# for SCENE in "${SCENE_LIST[@]}"
# do
#     mv outputs/${SCENE}/key_gaussian/${EXP_NAME} outputs/${EXP_NAME}/${SCENE}
#     rm -r outputs/${SCENE}
# done




SCENE_LIST=( "bench" "hydrant" "apple"   "skateboard" "teddybear"  )

EXP_NAME="CO3D_3DGS"
for SCENE in "${SCENE_LIST[@]}"
do
mkdir -p outputs/${SCENE}/key_gaussian/${EXP_NAME}/

    ns-train key_gaussian --data dataset/CFGS_Co3d/${SCENE}/${SCENE} \
    --optimizers.camera-opt.optimizer.lr 1e-3 \
    --optimizers.camera-opt.scheduler.lr-final 1e-5  --viewer.quit_on_train_completion True \
    --vis tensorboard \
    --timestamp ${EXP_NAME} \
    --pipeline.model.sh-degree-interval 1000 \
    --optimizers.camera-opt.scheduler.warmup-steps 1000 \
    --pipeline.model.cull-alpha-thresh 0.05 \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.densify-grad-thresh 0.0004 \
    --optimizers.camera-opt.scheduler.lr-pre-warmup 1e-5 --pipeline.model.use-abs-grad False \
    --pipeline.model.background-color black \
    --pipeline.model.split-screen-size 0.05 \
    --pipeline.model.cull-scale-thresh 0.05 \
    --pipeline.model.stop-screen-size-at 4000 \
    --pipeline.model.use-scale-regularization True \
    colmap --colmap-path sparse/0 --auto-scale-poses True \
    | tee -a outputs/${SCENE}/key_gaussian/${EXP_NAME}/train_log.txt 
    # export camera pose
    ns-export cameras --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-dir outputs/${SCENE}/key_gaussian/${EXP_NAME}/ 
    ns-eval --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml --output-path outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval.json --render-output-path outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval/  | tee -a outputs/${SCENE}/key_gaussian/${EXP_NAME}/eval_log.txt 
    ns-render interpolate  --load-config outputs/${SCENE}/key_gaussian/${EXP_NAME}/config.yml
    mv train_temp outputs/${SCENE}/key_gaussian/${EXP_NAME}/
    mv renders/output.mp4 outputs/${SCENE}/key_gaussian/${EXP_NAME}/output.mp4
done

# # move to the final directory 
mkdir -p outputs/${EXP_NAME}
for SCENE in "${SCENE_LIST[@]}"
do
    mv outputs/${SCENE}/key_gaussian/${EXP_NAME} outputs/${EXP_NAME}/${SCENE}
    rm -r outputs/${SCENE}
done
