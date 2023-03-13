python tools/main_ood.py \
-expID OODSwinT \
-task cls \
-arch swin_t \
-trainBatch 128 \
-LR 0.01 \
-dropLR 20 \
-GPU 0 \
> results/$(date +%F)_swint_train.txt

ALL_DISRUPTIONS=("context" "pose" "shape" "texture" "weather")
for DISRUPTION in "${ALL_DISRUPTIONS[@]}"
do
    echo "Running $DISRUPTION"
    python tools/main_ood.py \
        -expID OODSwinT \
        -task cls \
        -arch swin_t \
        -GPU 0 -test \
        -loadModel /misc/lmbraid21/jesslen/StarMap/exp/OODSwinT/model_cpu.pth \
        -nuisance $DISRUPTION
    python tools/EvalCls_OOD.py \
        --load_classification_dir /misc/lmbraid21/jesslen/StarMap/exp/OODSwinTTEST/classification_preds_$DISRUPTION.json \
        --load_pred_dir /misc/lmbraid21/jesslen/StarMap/exp/OODSwinTTEST/pose_preds_$DISRUPTION.pth \
        --nuisance $DISRUPTION \
    > results/$(date +%F)_swint_OOD_$DISRUPTION.txt
done


