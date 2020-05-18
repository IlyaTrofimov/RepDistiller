distill=rkd
gpu=$1

for part in 9 27 3 ; do
    for i in {0..99} ; do
        echo $part $i;
        python train_student.py --gpu $gpu --part $part --distill $distill --model_s MobileNetV2Trofim -r 0 -a 0 -b 1 --epochs 100 --arc $i --prefix $distill/part_$part;
    done
done
