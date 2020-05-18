from_arc=$1
to_arc=$2

cd ..

for (( arc = $from_arc; arc <= $to_arc; arc++ )); do
    python train_student.py --dataset imagenet --batch_size 128 --part 1 --model_s MobileNetV2Trofim --prefix imagenet/nokd/part_1 -r 1 -a 0 -b 0 --epochs 200 --arc $arc --val_freq 10 --num_workers 16
done
