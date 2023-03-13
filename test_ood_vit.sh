python tools/main_ood.py \
-inputRes 224 \
-expID OODViT \
-task cls \
-arch vit_b_16 \
-trainBatch 128 \
-LR 0.01 \
-dropLR 20 \
-GPU 0 \
> results/$(date +%F)_convnext_train.txt

ALL_DISRUPTIONS=("context" "pose" "shape" "texture" "weather")
for DISRUPTION in "${ALL_DISRUPTIONS[@]}"
do
    echo "Running $DISRUPTION"
    python tools/main_ood.py \
        -inputRes 224 \
        -expID OODViT \
        -task cls \
        -arch vit_b_16 \
        -GPU 0 -test \
        -loadModel /misc/lmbraid21/jesslen/StarMap/exp/OODViT/model_cpu.pth \
        -nuisance $DISRUPTION
    python tools/EvalCls_OOD.py \
        --load_classification_dir /misc/lmbraid21/jesslen/StarMap/exp/OODViTTEST/classification_preds_$DISRUPTION.json \
        --load_pred_dir /misc/lmbraid21/jesslen/StarMap/exp/OODViTTEST/pose_preds_$DISRUPTION.pth \
        --nuisance $DISRUPTION \
    > results/$(date +%F)_vit_OOD_$DISRUPTION.txt
done


