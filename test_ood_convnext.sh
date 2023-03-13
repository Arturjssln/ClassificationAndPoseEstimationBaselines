python tools/main_ood.py \
-expID OODConvNext \
-task cls \
-arch convnext \
-trainBatch 128 \
-LR 0.01 \
-dropLR 20 \
-GPU 0 \
> results/$(date +%F)_ood_convnext_train.txt

ALL_DISRUPTIONS=("context" "pose" "shape" "texture" "weather")
for DISRUPTION in "${ALL_DISRUPTIONS[@]}"
do
    echo "Running $DISRUPTION"
    python tools/main_ood.py \
        -expID OODConvNext \
        -task cls \
        -arch convnext \
        -GPU 0 -test \
        -loadModel /misc/lmbraid21/jesslen/StarMap/exp/OODConvNext/model_cpu.pth \
        -nuisance $DISRUPTION
    python tools/EvalCls_OOD.py \
        --load_classification_dir /misc/lmbraid21/jesslen/StarMap/exp/OODConvNextTEST/classification_preds_$DISRUPTION.json \
        --load_pred_dir /misc/lmbraid21/jesslen/StarMap/exp/OODConvNextTEST/pose_preds_$DISRUPTION.pth \
        --nuisance $DISRUPTION \
    > results/$(date +%F)_convnext_OOD_$DISRUPTION.txt
done


