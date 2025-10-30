#!/bin/bash
# run_demo_launch.sh — 两节点 1 卡 测试（使用 paddle.distributed.launch）
NODE_IP="$(hostname -I | awk '{print $1}')"
NODE0_IP="10.1.114.11"
NODE1_IP="10.1.114.20"

export GLOG_v=3
export NCCL_DEBUG=INFO
export PADDLE_XCCL_BACKEND=iluvatar_gpu

export PADDLE_TRAINER_ENDPOINTS="${NODE0_IP}:43576,${NODE1_IP}:43576"

if [ "${NODE_IP}" = "${NODE0_IP}" ]; then
  export PADDLE_TRAINER_ID=0
  export PADDLE_CURRENT_ENDPOINT="${NODE0_IP}:43576"
elif [ "${NODE_IP}" = "${NODE1_IP}" ]; then
  export PADDLE_TRAINER_ID=1
  export PADDLE_CURRENT_ENDPOINT="${NODE1_IP}:43576"
else
  echo "未识别当前节点 IP(${NODE_IP})，请手动设置 PADDLE_TRAINER_ID & PADDLE_CURRENT_ENDPOINT"
  exit 1
fi

echo "Starting demo on ${NODE_IP} as trainer ${PADDLE_TRAINER_ID}"
python3 -m paddle.distributed.launch --devices=0 demo.py
