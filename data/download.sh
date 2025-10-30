export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type dataset --resume-download ceval/ceval-exam   --local-dir ./ceval --local-dir-use-symlinks False
