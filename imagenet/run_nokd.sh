distill=nokd
gpu=$1
from_arc=$2
to_arc=$3

cd ..

for part in 27; do
    for (( arc = $from_arc; arc <= $to_arc; arc++ )); do
        python train_student3.py --distill kd --dataset imagenet --batch_size 128 --model_s MobileNetV2Trofim --gpu $gpu --part $part --prefix imagenet/$distill/part_$part -r 1 -a 0 -b 0 --epochs 100 --arc $arc --val_freq 10
    done
done
