for i in {0..4..1}
  do
    mkdir -p "ABLATION_RESULTS/$2/$1/set${i}"
    python3 DR-TANet-lightning/eval.py -i ~/SCD/.aim/ -e "$1" --configs DR-TANet-lightning/config/ -d "$2" -n "${i}"
    mv ABLATION_RESULTS/pred_* "ABLATION_RESULTS/GSV/$1/set${i}"

 done
