distill=nst
gpu=$1

cd ..

for part in 27 ; do
    for arc in 5; do
        python train_student.py --distill $distill --gpu $gpu --part $part --model_s ShuffleV2 --path_t shufflev2_teacher.pth --prefix shuffle/$distill/part_$part -r 1 -a 0 -b 12.5 --epochs 100 --arc $arc --val_freq 10
    done
done
