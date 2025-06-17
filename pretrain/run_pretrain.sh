
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export Flags_mp_aysnc_allreduce=1
export Flags_skip_mp_c_identity=1
export FLAGS_shard_norm_align_dp=0
export FLAGS_shard_use_reduce=1
export FLAGS_sync_before_allreduce=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --ips=192.168.0.1,192.168.0.2,192.168.0.3,192.168.0.4,192.168.0.5,192.168.0.6,192.168.0.7 run_pretrain.py ./config.json