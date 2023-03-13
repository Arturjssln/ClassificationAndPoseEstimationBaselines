ALL_DISRUPTIONS=("context" "pose" "shape" "texture" "weather")
for DISRUPTION in "${ALL_DISRUPTIONS[@]}"
do
    echo "Running $DISRUPTION"
    python tools/main_ood.py \
        -expID OODConvNext \
        -task cls \
        -arch convnext \
        -GPU 0 -test \
        -loadModel /misc/lmbraid21/jesslen/StarMap/exp/testConvNextOOD/model_cpu.pth \
        -nuisance $DISRUPTION
    python tools/EvalCls_OOD.py \
        --load_classification_dir /misc/lmbraid21/jesslen/StarMap/exp/OODConvNextTEST/classification_preds_$DISRUPTION.json \
        --load_pred_dir /misc/lmbraid21/jesslen/StarMap/exp/OODConvNextTEST/pose_preds_$DISRUPTION.pth \
        --nuisance $DISRUPTION \
    > results/$(date +%F)_OOD_convnext_$DISRUPTION.txt
done

for DISRUPTION in "${ALL_DISRUPTIONS[@]}"
do
    echo "Running $DISRUPTION"
    python tools/main_ood.py \
        -expID ResNetOOD \
        -task cls \
        -arch resnetext \
        -GPU 0 -test \
        -loadModel /misc/lmbraid21/jesslen/StarMap/exp/ResNetOOD/model_cpu.pth \
        -nuisance $DISRUPTION
    python tools/EvalCls_OOD.py \
        --load_classification_dir /misc/lmbraid21/jesslen/StarMap/exp/ResNetOODTEST/classification_preds_$DISRUPTION.json \
        --load_pred_dir /misc/lmbraid21/jesslen/StarMap/exp/ResNetOODTEST/pose_preds_$DISRUPTION.pth \
        --nuisance $DISRUPTION \
    > results/$(date +%F)_OOD_resnet_$DISRUPTION.txt
done

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
    > results/$(date +%F)_OOD_swint_$DISRUPTION.txt
done

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
    > results/$(date +%F)_OOD_vit_$DISRUPTION.txt
done






