distill=pkt
gpu=$1

for part in 3 ; do
    for i in {9..9} ; do
        echo $part $i;
        python train_student.py --gpu $gpu --part $part --distill $distill --model_s MobileNetV2Trofim -r 1 -a 0 -b 48e4 --epochs 100 --arc $i --prefix $distill/part_$part;
    done
done
