# ================================================================
# 🌐 WSL2 + ClashVergeRev 网络配置   # ~/.zshrc
# ================================================================

# ----------------------------------------------------------------
# 核心配置（按实际修改这里）
# ----------------------------------------------------------------
_PROXY_HOST="127.0.0.1"
_PROXY_MIXED_PORT="7897"        # ClashVergeRev mixed-port
#_PROXY_REDIR_PORT="7892"       # redir-port
_CLASH_API_PORT="9097"          # Clash API端口（按实际填）

_PROXY_HTTP="http://${_PROXY_HOST}:${_PROXY_MIXED_PORT}"
_PROXY_SOCKS="socks5://${_PROXY_HOST}:${_PROXY_MIXED_PORT}"
_NO_PROXY="localhost,127.0.0.1,::1,*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# ----------------------------------------------------------------
# 代理开关
# ----------------------------------------------------------------
proxy_on() {
    export http_proxy="${_PROXY_HTTP}"
    export https_proxy="${_PROXY_HTTP}"
    export all_proxy="${_PROXY_SOCKS}"
    export HTTP_PROXY="${_PROXY_HTTP}"
    export HTTPS_PROXY="${_PROXY_HTTP}"
    export ALL_PROXY="${_PROXY_SOCKS}"
    export no_proxy="${_NO_PROXY}"
    export NO_PROXY="${_NO_PROXY}"
    echo "🟢 Proxy ON → ${_PROXY_HTTP}"
}

proxy_off() {
    unset http_proxy https_proxy all_proxy no_proxy
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
    echo "🔴 Proxy OFF"
}

proxy_status() {
    if [[ -n "${http_proxy}" ]]; then
        echo "🟢 Proxy ON → ${http_proxy}"
    else
        echo "🔴 Proxy OFF"
    fi
}

# 单次代理执行，不污染全局
proxy_run() {
    http_proxy="${_PROXY_HTTP}"   \
    https_proxy="${_PROXY_HTTP}"  \
    all_proxy="${_PROXY_SOCKS}"   \
    HTTP_PROXY="${_PROXY_HTTP}"   \
    HTTPS_PROXY="${_PROXY_HTTP}"  \
    ALL_PROXY="${_PROXY_SOCKS}"   \
    "$@"
}

# sudo场景保留代理
proxysudo() {
    sudo -E env \
        http_proxy="${_PROXY_HTTP}"  \
        https_proxy="${_PROXY_HTTP}" \
        all_proxy="${_PROXY_SOCKS}"  \
        "$@"
}



# ----------------------------------------------------------------
# Prompt右侧显示代理状态
# ----------------------------------------------------------------
_proxy_prompt() {
    [[ -n "${http_proxy}" ]] && echo "%F{green}[proxy]%f" || echo "%F{red}[direct]%f"
}
RPROMPT='$(_proxy_prompt)'

# ================================================================
# 🔍 网络诊断
# ================================================================

# 内部：测试URL连通性，支持指定代理或直连
# 用法: _net_test <名称> <URL> [proxy|direct]
_net_test() {
    local name="$1" url="$2" mode="${3:-proxy}"
    local curl_opts=(-o /dev/null -s -w "%{http_code} %{time_total}"
                     --connect-timeout 6 --max-time 10)

    if [[ "$mode" == "direct" ]]; then
        curl_opts+=(--noproxy "*")
    else
        curl_opts+=(--proxy "${_PROXY_HTTP}")
    fi

    local out
    out=$(curl "${curl_opts[@]}" "$url" 2>/dev/null)
    local code="${out%% *}"
    local time="${out##* }"
    # 保留2位小数
    time=$(printf "%.2f" "$time")

    if [[ "$code" =~ ^(200|201|204|301|302|303|403|405)$ ]]; then
        printf "  🟢 %-18s HTTP %-3s  %ss\n" "${name}" "${code}" "${time}"
    elif [[ "$code" == "000" ]]; then
        printf "  🔴 %-18s 连接失败 (timeout/reset)\n" "${name}"
    else
        printf "  🟡 %-18s HTTP %-3s  %ss\n" "${name}" "${code}" "${time}"
    fi
}

# ----------------------------------------------------------------
# net_check：一键诊断
# ----------------------------------------------------------------
net_check() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 网络诊断  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. 代理状态
    echo "\n📡 【代理环境变量】"
    proxy_status

    # 2. Clash核心状态
    echo "\n⚙️  【Clash 核心】"
    local clash_ok=false
    # 自动探测API端口（尝试常用端口）
    local api_port=""
    for p in "${_CLASH_API_PORT}" 9090 9097 9098; do
        if curl -s --connect-timeout 1 "http://127.0.0.1:${p}/configs" > /dev/null 2>&1; then
            api_port="$p"
            clash_ok=true
            break
        fi
    done

    if $clash_ok; then
        echo "  🟢 Clash API 正常 → 端口 ${api_port}"
        local mode
        mode=$(curl -s "http://127.0.0.1:${api_port}/configs" 2>/dev/null \
               | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('mode','?'))" 2>/dev/null)
        echo "  📋 运行模式: ${mode}"
    else
        echo "  🔴 Clash API 不可达（检查ClashVergeRev是否运行）"
    fi

    # 3. DNS解析检查
    echo "\n🌐 【DNS 解析】"
    # 检测可用DNS端口
    local dns_port=""
    for p in 1053 1054 7874; do
        if dig +short +timeout=2 google.com @127.0.0.1 -p "$p" > /dev/null 2>&1; then
            dns_port="$p"
            break
        fi
    done

    if [[ -n "$dns_port" ]]; then
        local cn_ip gfw_ip
        cn_ip=$(dig +short +timeout=3 mirrors.tuna.tsinghua.edu.cn @127.0.0.1 -p "$dns_port" 2>/dev/null | grep -v '\.$' | head -1)
        gfw_ip=$(dig +short +timeout=3 google.com @127.0.0.1 -p "$dns_port" 2>/dev/null | head -1)

        if [[ "$cn_ip" == 198.18.* ]]; then
            echo "  🔴 国内镜像 → ${cn_ip} (Fake IP！fake-ip-filter未生效)"
        else
            echo "  🟢 国内镜像 → ${cn_ip:-解析失败}"
        fi

        if [[ "$gfw_ip" == 198.18.* ]]; then
            echo "  🟢 国外域名 → ${gfw_ip} (Fake IP，TUN正常处理)"
        else
            echo "  🟡 国外域名 → ${gfw_ip:-解析失败} (返回真实IP)"
        fi
    else
        echo "  🟡 Clash DNS端口未找到，使用系统DNS"
        local sys_cn sys_gfw
        sys_cn=$(dig +short +timeout=3 mirrors.tuna.tsinghua.edu.cn 2>/dev/null | grep -v '\.$' | head -1)
        sys_gfw=$(dig +short +timeout=3 google.com 2>/dev/null | head -1)
        echo "  国内镜像 → ${sys_cn:-失败}"
        echo "  国外域名 → ${sys_gfw:-失败}"
    fi

    # 4. 国内镜像连通性（走代理，让Clash判断直连）
    echo "\n🇨🇳 【国内镜像连通性】(经7897→Clash→DIRECT)"
    _net_test "清华Ubuntu源"    "https://mirrors.tuna.tsinghua.edu.cn" proxy
    _net_test "阿里镜像源"      "https://mirrors.aliyun.com"           proxy
    _net_test "清华PyPI"        "https://pypi.tuna.tsinghua.edu.cn"    proxy
    _net_test "npm镜像"         "https://registry.npmmirror.com"       proxy

    # 5. 国外站点连通性（走代理）
    echo "\n🌍 【国外站点连通性】(经7897→Clash→代理节点)"
    _net_test "Google"          "https://www.google.com"     proxy
    _net_test "GitHub"          "https://github.com"         proxy
    _net_test "HuggingFace"     "https://huggingface.co"     proxy
    _net_test "OpenAI API"      "https://api.openai.com"     proxy
    _net_test "Cloudflare"      "https://cloudflare.com"     proxy

    echo "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💡 提示: 国内失败→检查Clash规则; 国外失败→检查节点选择"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ----------------------------------------------------------------
# net_speed：网速测试
# ----------------------------------------------------------------
net_speed() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚡ 网速测试  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    _speed_test() {
        local label="$1" url="$2" proxy="$3"
        local opts=(-o /dev/null -s -w "%{speed_download}"
                    --connect-timeout 5 --max-time 15)
        [[ "$proxy" == "yes" ]] && opts+=(--proxy "${_PROXY_HTTP}") || opts+=(--noproxy "*")
        local bytes
        bytes=$(curl "${opts[@]}" "$url" 2>/dev/null)
        printf "  %-20s %s MB/s\n" "${label}" \
            "$(awk "BEGIN{printf \"%.2f\", ${bytes:-0}/1024/1024}")"
    }

    echo "\n🇨🇳 国内（经代理→Clash→DIRECT）："
    _speed_test "清华源" \
        "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ls-lR.gz" yes

    echo "\n🌍 代理出口："
    _speed_test "Cloudflare" \
        "https://speed.cloudflare.com/__down?bytes=10000000" yes

    echo "\n🌐 当前出口IP："
    printf "  %-20s" "经代理出口："
    curl -s --proxy "${_PROXY_HTTP}" --connect-timeout 5 \
        "https://api.ipify.org" 2>/dev/null || echo "获取失败"
    echo

    echo "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ----------------------------------------------------------------
# clash_check：Clash配置诊断
# ----------------------------------------------------------------
clash_check() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚙️  Clash 配置诊断"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 自动探测API端口
    local api_port=""
    for p in "${_CLASH_API_PORT}" 9090 9097 9098; do
        if curl -s --connect-timeout 1 "http://127.0.0.1:${p}/configs" > /dev/null 2>&1; then
            api_port="$p"; break
        fi
    done

    [[ -z "$api_port" ]] && echo "❌ Clash API不可达" && return 1

    curl -s "http://127.0.0.1:${api_port}/configs" 2>/dev/null | python3 -c "
import sys, json

d = json.load(sys.stdin)

# TUN
tun = d.get('tun', {})
print('\n【TUN 配置】')
print(f'  启用:     {tun.get(\"enable\")}')
print(f'  协议栈:   {tun.get(\"stack\")}')
print(f'  DNS劫持:  {tun.get(\"dns-hijack\")}')
print(f'  IPv4:     {tun.get(\"inet4-address\")}')
print(f'  IPv6:     {tun.get(\"inet6-address\", \"❌ 未配置\")}')

# 端口
print('\n【端口配置】')
print(f'  mixed-port:  {d.get(\"mixed-port\", \"未配置\")}')
print(f'  redir-port:  {d.get(\"redir-port\", \"未配置\")}')

# DNS
dns = d.get('dns', {})
print('\n【DNS 配置】')
if not dns:
    print('  ❌ 未加载（检查ClashVergeRev覆写配置）')
else:
    status = lambda v, ok='✅', fail='❌': ok if v else fail
    print(f'  模式:              {dns.get(\"enhanced-mode\")}')
    print(f'  监听:              {dns.get(\"listen\")}')
    print(f'  respect-rules:     {status(dns.get(\"respect-rules\"))} {dns.get(\"respect-rules\", False)}')
    ns = dns.get('nameserver', [])
    print(f'  nameserver:')
    for n in ns: print(f'    - {n}')
    pns = dns.get('proxy-server-nameserver', [])
    print(f'  proxy-server-ns:   {\"✅\" if pns else \"❌ 未配置\"}')
    for n in pns: print(f'    - {n}')
    ff = dns.get('fake-ip-filter', [])
    print(f'  fake-ip-filter:    {len(ff)}条')
    for f in ff: print(f'    - {f}')

# 模式和节点
print(f'\n【运行状态】')
print(f'  模式: {d.get(\"mode\")}')
"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ================================================================
# 🛠️ 常用工具封装（自动走代理）
# ================================================================

# apt：proxysudo确保sudo下也有代理
proxy_apt() { proxysudo apt-get "$@" };

# pip：通过环境变量传代理（兼容pip/uv）
proxy_pip()  { proxy_run command pip  "$@" }
proxy_uv()   { proxy_run command uv   "$@" }

# wget：通过-e传递代理
proxy_wget() {
    command wget \
        -e "http_proxy=${_PROXY_HTTP}"  \
        -e "https_proxy=${_PROXY_HTTP}" \
        "$@"
}


# ================================================================
# 📖 使用说明
# ================================================================
net_help() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📖 网络工具  WSL2 + ClashVergeRev"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  诊断命令:"
    echo "    net_check        一键检查所有网络状态"
    echo "    net_speed        测速 + 出口IP"
    echo "    clash_check      Clash配置详情"
    echo ""
    echo "  代理控制:"
    echo "    proxy_on         开启全局代理"
    echo "    proxy_off        关闭全局代理"
    echo "    proxy_status     查看状态"
    echo "    proxy_run <cmd>  单次代理执行（不污染全局）"
    echo "    proxysudo <cmd>  sudo场景走代理"
    echo ""
    echo "  已封装（自动走代理）:"
    echo "    proxy_apt  proxy_pip   proxy_uv  proxy_wget"
    echo ""
    echo "  代理地址: ${_PROXY_HTTP}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}






