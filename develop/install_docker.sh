#!/bin/bash
set -e

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始安装 Docker 和 Docker Compose...${NC}"

# 检查 root 权限
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}请使用 root 用户或 sudo 运行此脚本${NC}"
    exit 1
fi

# 检测系统版本
if ! grep -q 'CentOS Linux 8' /etc/os-release; then
    echo -e "${RED}此脚本仅适用于 CentOS Linux 8${NC}"
    exit 1
fi

# 卸载旧版本 Docker
echo -e "${GREEN}卸载旧版本 Docker...${NC}"
yum remove -y docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine

# 安装必要工具
echo -e "${GREEN}安装必要工具...${NC}"
yum install -y yum-utils curl wget

# 备份原有 YUM 仓库
echo -e "${GREEN}备份原有 YUM 仓库...${NC}"
mkdir -p /etc/yum.repos.d/backup
mv /etc/yum.repos.d/CentOS-*.repo /etc/yum.repos.d/backup/ 2>/dev/null || true

# 配置阿里云 CentOS 8 基础源
echo -e "${GREEN}配置阿里云 CentOS 8 基础源...${NC}"
curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-8.repo
sed -i 's/$releasever/8/g' /etc/yum.repos.d/CentOS-Base.repo

# 添加阿里云 Docker CE 仓库
echo -e "${GREEN}添加阿里云 Docker CE 仓库...${NC}"
yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

# 更新缓存
echo -e "${GREEN}更新 YUM 缓存...${NC}"
yum makecache

# 安装 Docker
echo -e "${GREEN}安装 Docker CE...${NC}"
yum install -y docker-ce docker-ce-cli containerd.io

# 启动 Docker
echo -e "${GREEN}启动 Docker 服务...${NC}"
systemctl enable docker
systemctl start docker

# 验证 Docker
docker --version

# 安装 Docker Compose
echo -e "${GREEN}安装 Docker Compose...${NC}"

# 定义 Docker Compose 版本（可根据需要修改）
COMPOSE_VERSION="v2.24.0"
COMPOSE_BINARY_URL="https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64"

# 尝试下载 Docker Compose 二进制
if curl -L "${COMPOSE_BINARY_URL}" -o /usr/local/bin/docker-compose; then
    chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose 二进制安装成功${NC}"
else
    echo -e "${RED}下载 Docker Compose 二进制失败，尝试安装插件版本...${NC}"
    # 安装 docker-compose-plugin
    yum install -y docker-compose-plugin
    # 创建包装脚本，使 docker-compose 命令可用
    cat > /usr/local/bin/docker-compose << 'EOF'
#!/bin/bash
docker compose "$@"
EOF
    chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose 插件安装成功（通过包装脚本）${NC}"
fi

# 验证 Docker Compose
docker-compose --version

echo -e "${GREEN}安装完成！${NC}"