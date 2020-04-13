for arc in {839..1000}; do
    sbatch run_zhores.sh --distill kd --gpu 0 --part 1 --model_s ShuffleV2 --path_t /gpfs/gpfs0/i.trofimov/nas/shufflev2_teacher.pth --prefix base/part1 -r 1 -a 0 -b 0 --epochs 100 --arc $arc --arcs_dir /gpfs/gpfs0/i.trofimov/nas --res_dir /gpfs/gpfs0/i.trofimov/nas/result
done
