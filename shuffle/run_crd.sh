distill=crd
gpu=$1

cd ..

for part in 27 9 3; do
    for arc in {0..299}; do
        python train_student.py --distill $distill --gpu $gpu --part $part --model_s ShuffleV2 --path_t shufflev2_teacher.pth --prefix shuffle/$distill/part_$part -r 1 -a 0 --nce_k 4096 --nce_t 0.05 -b 1 --epochs 100 --arc $arc --val_freq 10
    done
done
