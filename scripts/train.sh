# # Run full finetuning vision experiments (note: covariances only needed for regmean)
# for method in isoc regmean eigcov; do
#     python scripts/vision/eval_task_addition.py \
#     --model=ViT-B-16 \
#     --finetuning-mode=standard  \
#     --merge-func="$method" \
#     --mha=split \
#     --cov-dir=results/ViT-B-16/covariances_strain_n10_b32_tsm_attnsplit_efull_ftlora \
#     --results-db=results/resuts.csv
# done


# Run LoRA vision experiments
for method in isoc regmean eigcov; do
    python scripts/vision/eval_task_addition.py \
    --model=ViT-B-16 \
    --finetuning-mode=lora  \
    --merge-func="$method" \
    --mha=split \
    --cov-dir=results/ViT-B-16/covariances_strain_n10_b32_tsm_attnsplit_efull_ftlora \
    --results-db=results/resuts.csv
done