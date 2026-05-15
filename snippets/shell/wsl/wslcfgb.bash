# ================================================================
# 🌐 WSL2 (Mirrored) + ClashVergeRev 极简网络工具箱
# ================================================================

# ----------------------------------------------------------------
# 1. 核心地址配置
# ----------------------------------------------------------------
_PROXY_HOST="127.0.0.1"
_PROXY_PORT="7897"             # ClashVergeRev 代理流量端口 mixed_port
_CLASH_API_PORT="6553"         # ClashVergeRev API/控制面板端口 ()

_PROXY_HTTP="http://${_PROXY_HOST}:${_PROXY_PORT}"
_PROXY_SOCKS="socks5://${_PROXY_HOST}:${_PROXY_PORT}"
_NO_PROXY="localhost,127.0.0.1,::1,*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# ----------------------------------------------------------------
# 2. 环境变量代理控制
# ----------------------------------------------------------------
proxy_on() {
    export http_proxy="${_PROXY_HTTP}" https_proxy="${_PROXY_HTTP}" all_proxy="${_PROXY_SOCKS}"
    export HTTP_PROXY="${_PROXY_HTTP}" HTTPS_PROXY="${_PROXY_HTTP}" ALL_PROXY="${_PROXY_SOCKS}"
    export no_proxy="${_NO_PROXY}" NO_PROXY="${_NO_PROXY}"
    [[ "$1" != "-q" ]] && echo "🟢 环境变量代理已开启 → ${_PROXY_HTTP}"
}
proxy_off() {
    unset http_proxy https_proxy all_proxy no_proxy
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
    echo "🔴 环境变量代理已关闭"
}
proxy_status() {
    [[ -n "${http_proxy}" ]] && echo "🟢 代理环境变量: 生效中" || echo "🔴 代理环境变量: 未生效 (纯 TUN 接管)"
}
proxy_run() {
    http_proxy="${_PROXY_HTTP}" https_proxy="${_PROXY_HTTP}" all_proxy="${_PROXY_SOCKS}" "$@"
}
proxysudo() {
    sudo -E env http_proxy="${_PROXY_HTTP}" https_proxy="${_PROXY_HTTP}" all_proxy="${_PROXY_SOCKS}" "$@"
}

# ----------------------------------------------------------------
# 3. 常用保底指令封装 (Alias)
# ----------------------------------------------------------------
alias proxy_apt='proxysudo apt-get'
alias proxy_pip='proxy_run command pip'
alias proxy_uv='proxy_run command uv'
alias proxy_wget='command wget -e "http_proxy=${_PROXY_HTTP}" -e "https_proxy=${_PROXY_HTTP}"'

# ================================================================
# 🔍 极简网络诊断面板
# ================================================================
_net_test() {
    local name="$1" url="$2"
    local out code time
    out=$(curl -o /dev/null -s -w "%{http_code}:%{time_total}" --connect-timeout 3 --max-time 5 "$url" 2>/dev/null || echo "000:0.00")
    code="${out%%:*}"
    time=$(printf "%.2f" "${out##*:}")

    if [[ "$code" =~ ^(200|201|204|301|302|303|403|405)$ ]]; then
        echo | awk -v n="$name" -v c="$code" -v t="${time}s" '{printf "  🟢 %-14s HTTP %-3s (%s)\n", n, c, t}'
    else
        echo | awk -v n="$name" -v c="$code" '{printf "  🔴 %-14s 失败 (HTTP %s)\n", n, c}'
    fi
}
net_check() {
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " 🔍 网络连通性测试 | $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -e "\n📡 【当前环境状态】"
    proxy_status

    echo -e "\n🇨🇳 【国内节点测试】"
    _net_test "清华 Ubuntu 源" "https://mirrors.tuna.tsinghua.edu.cn"
    _net_test "阿里 镜像源"    "https://mirrors.aliyun.com"

    echo -e "\n🌍 【海外节点测试】"
    _net_test "Google"         "https://www.google.com"
    _net_test "GitHub"         "https://github.com"
    _net_test "HuggingFace"    "https://huggingface.co"
    _net_test "OpenAI API"     "https://api.openai.com"

    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
}
net_speed() {
    echo -e "\n⚡ 终端出口测速"
    echo -n "  📍 当前出口 IP : "
    curl -s --connect-timeout 3 "https://api.ipify.org" 2>/dev/null || echo "获取失败"
    echo ""

    echo "  🚀 Cloudflare (10MB) 下载测试..."
    curl -o /dev/null -s -w "  📊 最终速度: %{speed_download} bytes/s\n" \
        --connect-timeout 5 --max-time 15 \
        "https://speed.cloudflare.com/__down?bytes=10000000" 2>/dev/null \
        | awk '{printf "  ✅ 测速完成: %.2f MB/s\n", $3/1024/1024}'
    echo ""
}
clash_check() {
    echo -e "\n⚙️  ClashVergeRev API 状态探测"
    # 优先尝试你设置的端口，失败则尝试其它常用端口
    local api_port=""
    for p in "${_CLASH_API_PORT}" 9097 9090; do
        if curl -s --connect-timeout 1 "http://${_PROXY_HOST}:${p}/configs" > /dev/null 2>&1; then
            api_port="$p"; break
        fi
    done

    if [[ -n "$api_port" ]]; then
        curl -s "http://${_PROXY_HOST}:${api_port}/configs" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'  ▶ API 端口: {sys.argv[1]}')
    print(f'  ▶ 运行模式: {d.get(\"mode\", \"未知\").upper()}')
    print(f'  ▶ TUN 状态: {\"✅ 开启\" if d.get(\"tun\", {}).get(\"enable\") else \"❌ 关闭\"}')
    print(f'  ▶ 局域网连: {\"✅ 允许\" if d.get(\"allow-lan\") else \"❌ 禁止\"}')
except Exception:
    print('  ❌ 数据解析失败')
" "$api_port"
    else
        echo "  ❌ API 不可达，请检查 ClashVergeRev 是否运行"
    fi
    echo ""
}
net_help() {
    echo -e "\n📖 WSL 网络工具箱"
    echo -e "  [检测] net_check   | net_speed   | clash_check"
    echo -e "  [控制] proxy_on    | proxy_off   | proxy_status"
    echo -e "  [保底] proxy_run   | proxysudo"
    echo -e "  [常用] proxy_apt   | proxy_pip   | proxy_uv | proxy_wget\n"
}