#!/bin/bash

set -e

echo "🧠 Gensyn RL-Swarm 一键安装脚本 for Linux（支持 GPU / CPU 模式）"

# Step 1: 安装依赖工具
echo "📦 安装基础工具..."
sudo apt update
sudo apt install -y git curl python3.10 python3.10-venv python3.10-dev wget build-essential

# Step 2: 安装 Node.js（18.x）和 Yarn（使用 corepack）
if ! command -v node &> /dev/null; then
  echo "🧱 安装 Node.js 18.x..."
  curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
  sudo apt install -y nodejs
fi

echo "🧵 启用 corepack + Yarn..."
corepack enable
corepack prepare yarn@stable --activate

# Step 3: 克隆 Gensyn 仓库
if [ ! -d "rl-swarm" ]; then
  echo "🔁 克隆 gensyn-ai/rl-swarm 仓库..."
  git clone https://github.com/gensyn-ai/rl-swarm.git
else
  echo "📁 rl-swarm 已存在，跳过克隆"
fi

cd rl-swarm

# Step 4: 创建虚拟环境
echo "🧪 创建 Python 虚拟环境..."
python3.10 -m venv .venv
source .venv/bin/activate

# Step 5: 安装 Python 依赖
echo "📦 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 6: 模式选择（CPU / GPU）
echo ""
echo "🧠 请选择运行模式："
echo "1) CPU-only（推荐，兼容性强）"
echo "2) GPU（NVIDIA CUDA）（需已安装 GPU 驱动 + CUDA）"
read -p "请输入选项编号 [默认 1]：" mode_choice

if [[ "$mode_choice" == "2" ]]; then
  echo "🚀 使用 GPU 模式（确保你已正确安装 NVIDIA 驱动 + CUDA）"
else
  echo "🛡️ 启用 CPU-only 模式"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  echo "🛠️ 替换代码中 device 设置为 CPU"
  sed -i 's/torch\.device("mps" if torch\.backends\.mps\.is_available() else "cpu")/torch.device("cpu")/g' hivemind_exp/trainer/hivemind_grpo_trainer.py
fi

# Step 7: 启动节点
echo "🚀 启动 RL-Swarm 节点..."
echo "⚠️ 当提示是否加入 testnet 时，请输入 Y 或直接按回车。"
sleep 2
bash run_rl_swarm.sh
