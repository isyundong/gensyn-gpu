#!/bin/bash

set -e

echo "🧠 Gensyn RL-Swarm 一键安装脚本 for Linux (支持 GPU / CPU 模式)"

# 确定当前 Ubuntu 版本
UBUNTU_VERSION=$(lsb_release -rs | cut -d. -f1)

# 根据 Ubuntu 版本选择对应 Python
if [[ "$UBUNTU_VERSION" == "20" ]]; then
  PYTHON_EXEC=python3.10
  PYTHON_PKG="python3.10 python3.10-venv python3.10-dev"
elif [[ "$UBUNTU_VERSION" == "22" ]]; then
  PYTHON_EXEC=python3.10
  PYTHON_PKG="python3.10 python3.10-venv python3.10-dev"\elif [[ "$UBUNTU_VERSION" == "24" ]]; then
  PYTHON_EXEC=python3
  PYTHON_PKG="python3 python3-venv python3-dev"
else
  echo "❌ 未支持的 Ubuntu 版本，请使用 20.04 / 22.04 / 24.04"
  exit 1
fi

# Step 1: 安装基础工具
sudo apt update
sudo apt install -y git curl wget build-essential $PYTHON_PKG

# Step 2: 安装 Node.js 18.x 和 Yarn
if ! command -v node &> /dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
  sudo apt install -y nodejs
fi

corepack enable
corepack prepare yarn@stable --activate

# Step 3: 克隆 Gensyn 仓库
if [ ! -d "rl-swarm" ]; then
  git clone https://github.com/gensyn-ai/rl-swarm.git
fi
cd rl-swarm

# Step 4: 创建虚拟环境
$PYTHON_EXEC -m venv .venv
source .venv/bin/activate

# Step 5: 安装 Python 依赖
pip install --upgrade pip
pip install -r requirements.txt

# Step 6: 请选择运行模式
echo "\n🧠 选择运行模式："
echo "1) CPU-only (符合性最好，稳定)"
echo "2) GPU (CUDA/MPS)"
read -p "输入选项 [默认 1]：" mode_choice

if [[ "$mode_choice" == "2" ]]; then
  echo "🚀 启用 GPU 模式"
else
  echo "🛡️ 启用 CPU-only 模式"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  sed -i 's/torch\.device("mps" if torch\.backends\.mps\.is_available() else "cpu")/torch.device("cpu")/g' hivemind_exp/trainer/hivemind_grpo_trainer.py
fi

# Step 7: 启动 RL Swarm 节点
echo "🚀 启动 RL-Swarm 节点..."
echo "⚠️ 当提示是否加入 testnet 时，请输入 Y 或直接回车"
sleep 2
bash run_rl_swarm.sh
