distill=attention
gpu=$1

cd ..

for part in 27 9 3; do
    for arc in {0..999}; do
        python train_student.py --distill $distill --kd_T 32 --gpu $gpu --part $part --model_s ShuffleV2 --path_t shufflev2_teacher.pth --prefix shuffle/$distill/part_$part -r 1 -a 0 -b 1e3 --epochs 100 --arc $arc --val_freq 10
    done
done
