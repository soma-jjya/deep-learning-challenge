#!/bin/bash
# EC2 user-data: 인스턴스 최초 부팅 시 root로 자동 실행된다.
# 무거운 파이썬 설치는 여기서 하지 않는다(부팅 지연·실패 위험) — setup_env.sh로 분리.
set -x
exec > /var/log/bootstrap.log 2>&1

apt-get update -y
apt-get install -y tmux git htop

# Node 20 + Claude Code
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# uv (파이썬 패키지 매니저) — ubuntu 사용자로 설치
sudo -u ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# 2단계 세팅 스크립트를 홈에 배치
cat > /home/ubuntu/setup_env.sh << 'EOF'
#!/bin/bash
# 서버 접속 후 1회 실행: bash ~/setup_env.sh
set -e
export PATH="$HOME/.local/bin:$PATH"

# private 레포라 GitHub 토큰 필요 (fine-grained PAT, Contents: Read and write)
if [ -z "$GITHUB_TOKEN" ]; then
  echo '먼저 GitHub 토큰을 등록하세요:'
  echo '  echo "export GITHUB_TOKEN=github_pat_..." >> ~/.bashrc && source ~/.bashrc'
  exit 1
fi
mkdir -p ~/work && cd ~/work
if [ ! -d deep-learning-challenge ]; then
  git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/soma-jjya/deep-learning-challenge.git"
fi
cd deep-learning-challenge

uv venv --python 3.11
# Unsloth가 torch·transformers 등 호환 버전을 함께 끌어온다
uv pip install unsloth trl datasets wandb "huggingface_hub[cli]" pandas vllm

# 유휴 30분 자동 정지 감시자 등록 (cron, 5분 간격)
chmod +x remote/watchdog_autostop.sh remote/run_experiments.sh
( crontab -l 2>/dev/null | grep -v watchdog_autostop ; \
  echo "*/5 * * * * NTFY_TOPIC=\$NTFY_TOPIC /home/ubuntu/work/deep-learning-challenge/remote/watchdog_autostop.sh" ) | crontab -

echo ''
echo '=== 세팅 완료. 다음 단계 ==='
echo '1) claude 토큰:   echo "export CLAUDE_CODE_OAUTH_TOKEN=..." >> ~/.bashrc'
echo '2) 폰 알림 주제:  echo "export NTFY_TOPIC=ajudl-원하는비밀문자열" >> ~/.bashrc && source ~/.bashrc'
echo '3) huggingface-cli login / wandb login'
echo '4) 러너 시작:     nohup bash remote/run_experiments.sh > runner.log 2>&1 &'
EOF
chmod +x /home/ubuntu/setup_env.sh
chown ubuntu:ubuntu /home/ubuntu/setup_env.sh

touch /home/ubuntu/BOOTSTRAP_DONE
chown ubuntu:ubuntu /home/ubuntu/BOOTSTRAP_DONE
