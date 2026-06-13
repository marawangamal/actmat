# Legacy Group Renames

The old language/T5 artifact groups were renamed so legacy outputs do not
collide with the new explicit-path `scripts/t5/` groups.

Applied under both `artifacts/checkpoints/{t5-base,t5-large}/` and
`artifacts/results/{t5-base,t5-large}/`:

- `group-main` -> `group-legacy-7`
- `group-headmean` -> `group-legacy-headmean`

`group-legacy-7` corresponds to the seven-task T5 suite:
`qasc,wiki_qa,quartz,paws,story_cloze,winogrande,wsc`.

New explicit T5 runs should use group names that encode fine-tuning mode and
task count, for example `group-fft-7` or `group-lora-7`.
