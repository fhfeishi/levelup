# Windows + Clash / WSL Ubuntu + Clash 配置 Pipeline

> 依据 `notes.md` 整理。本文把 Clash 统一指 Clash Verge Rev / mihomo。
> 当前机器验证时间：2026-05-28。

## 脚本速查

Windows：

```cmd
win_cvr.cmd help
win_cvr.cmd check 7897
win_cvr.cmd apply 7897
```

WSL Ubuntu-22.04：

```bash
bash wsl_cvr.sh help
bash wsl_cvr.sh check 7897
bash wsl_cvr.sh apply 7897
```

日志位置：

```text
Windows: win_cvr_report_*.log
WSL:     wsl_cvr_report_*.log
```

默认 `check` 只读，不改配置。`apply` 是保守配置：

```text
Windows apply:
  设置当前用户 System Proxy 到 127.0.0.1:<port>
  提高 IPv4 优先级
  flushdns
  不禁用网卡 IPv6，不修改 Clash 配置

WSL apply:
  添加 proxy_on/proxy_off/proxy_run/proxysudo 到 ~/.bashrc
  设置 git 代理到 127.0.0.1:<port>
  设置 /etc/gai.conf IPv4 优先
  设置 apt ForceIPv4
  不锁死 /etc/resolv.conf，不默认开启全局代理
```

## 0. 第一优先级：判断这台机器该优先 IPv4 还是 IPv6

在配置 Clash、WSL、pip、uv、apt 之前，先判断当前机器的 IPv6 是否真的可用。很多问题不是代理端口导致的，而是系统把一个不可用或半可用的 IPv6 路径放到了更高优先级。

典型坏场景：

```text
某张无线网卡启用了 IPv6。
Windows 认为这条 IPv6 路径优先级更高。
curl / pip / uv / apt / 浏览器优先尝试 IPv6。
但实际 IPv6 无法稳定访问外网或被 Clash/TUN/DNS 处理异常。
结果表现为超时、TLS handshake eof、连接 reset、apt 卡住。
```

### 0.1 Windows：先看网卡 IPv4 / IPv6 状态

PowerShell：

```powershell
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed

Get-NetAdapterBinding -Name "*" | Where-Object {
  $_.ComponentID -in "ms_tcpip", "ms_tcpip6"
} | Format-Table Name, DisplayName, ComponentID, Enabled
```

重点看当前正在联网的物理网卡，例如 WLAN、Wi-Fi、Ethernet：

```text
ms_tcpip  = IPv4
ms_tcpip6 = IPv6
```

如果某张当前联网的无线网卡启用了 IPv6，但后面的 `curl -6` 测试失败，这张网卡的 IPv6 就是高风险项。

### 0.2 Windows：用 nslookup 看 DNS 返回了哪些地址

`nslookup` 适合判断域名有没有 AAAA 记录，以及 DNS 是否被 Clash fake-ip 影响。

```powershell
nslookup pypi.org
nslookup files.pythonhosted.org
nslookup mirrors.aliyun.com
nslookup mirrors.tuna.tsinghua.edu.cn
nslookup github.com
nslookup google.com
nslookup developer.download.nvidia.com
nslookup astral.sh
```

判断：

```text
如果返回 Address 里有 IPv6/AAAA：
  说明程序可能尝试 IPv6。

如果返回 198.18.x.x：
  说明 Clash fake-ip/DNS 正在参与解析。

如果国内镜像返回 198.18.x.x：
  检查 fake-ip-filter 或规则，国内镜像更适合返回真实 IP。
```

`nslookup` 只能说明 DNS 返回结果，不能证明 HTTPS 一定可用。真正连通性要继续用 `curl -4/-6`。

### 0.3 Windows：用 curl 分别测试 IPv4 / IPv6

PowerShell：

```powershell
$targets = @(
  "https://pypi.org/simple/pillow/",
  "https://files.pythonhosted.org/",
  "https://mirrors.aliyun.com/pypi/simple/",
  "https://pypi.tuna.tsinghua.edu.cn/simple/",
  "https://mirrors.tuna.tsinghua.edu.cn/",
  "https://github.com/",
  "https://google.com/",
  "https://developer.download.nvidia.com/compute/cuda/repos/",
  "https://astral.sh/"
)

foreach ($u in $targets) {
  Write-Host "`n=== $u ==="
  curl.exe -4 -I --max-time 10 $u
  curl.exe -6 -I --max-time 10 $u
}
```

判断：

```text
curl -4 大多成功，curl -6 大多失败：
  当前机器应优先 IPv4，限制或降低 IPv6 优先级。

curl -4 和 curl -6 都成功：
  IPv6 可以保留，不需要急着禁用。

只有个别站点 curl -6 失败：
  优先按站点/工具处理，不要立刻全局禁 IPv6。

curl 默认失败，但 curl -4 成功：
  很可能系统默认优先用了坏 IPv6。
```

### 0.4 Windows：优先 IPv4，而不是一上来全局禁 IPv6

优先推荐调整前缀策略，让 Windows 更倾向 IPv4：

```powershell
netsh interface ipv6 show prefixpolicies
netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4
netsh interface ipv6 show prefixpolicies
```

预期是 `::ffff:0:0/96` 的优先级高于普通 IPv6 `::/0`。这样系统会更倾向 IPv4，但 IPv6 仍然保留。

### 0.5 Windows：确认坏 IPv6 时，禁用当前联网网卡的 IPv6

如果确认某张无线网卡的 IPv6 被优先使用且不可用，可以只禁用这张网卡的 IPv6：

```powershell
Get-NetAdapter
Disable-NetAdapterBinding -Name "Wi-Fi" -ComponentID ms_tcpip6
```

如果网卡名是中文，例如“WLAN”或“无线网络连接”，按实际名称替换：

```powershell
Disable-NetAdapterBinding -Name "WLAN" -ComponentID ms_tcpip6
```

恢复：

```powershell
Enable-NetAdapterBinding -Name "Wi-Fi" -ComponentID ms_tcpip6
```

注意：这一步优先级很高，但不要盲目执行。只有在 `curl -6` 明确失败，且系统确实优先使用这条坏 IPv6 时才做。

### 0.6 WSL/Ubuntu：同步判断 IPv4 / IPv6

WSL：

```bash
getent ahosts pypi.org | head
getent ahosts github.com | head
getent ahosts mirrors.aliyun.com | head
getent ahosts mirrors.tuna.tsinghua.edu.cn | head
getent ahosts developer.download.nvidia.com | head
getent ahosts astral.sh | head

curl -4 -I --max-time 10 https://pypi.org/simple/pillow/
curl -6 -I --max-time 10 https://pypi.org/simple/pillow/
curl -4 -I --max-time 10 https://github.com/
curl -6 -I --max-time 10 https://github.com/
curl -4 -I --max-time 10 https://mirrors.aliyun.com/pypi/simple/
curl -6 -I --max-time 10 https://mirrors.aliyun.com/pypi/simple/
curl -4 -I --max-time 10 https://developer.download.nvidia.com/compute/cuda/repos/
curl -6 -I --max-time 10 https://developer.download.nvidia.com/compute/cuda/repos/
curl -4 -I --max-time 10 https://astral.sh/
curl -6 -I --max-time 10 https://astral.sh/
```

如果 WSL 里 IPv6 不稳定，优先让 WSL 使用 IPv4：

```bash
sudo nano /etc/gai.conf
```

取消注释或添加：

```conf
precedence ::ffff:0:0/96  100
```

然后在 Windows PowerShell 重启 WSL：

```powershell
wsl --shutdown
```

### 0.7 apt / uv / pip 的 IPv4 策略

apt 可以直接强制 IPv4：

```bash
sudo apt-get -o Acquire::ForceIPv4=true update
```

如果这条稳定，而普通 `sudo apt-get update` 不稳定，再写入永久配置：

```bash
echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4
```

uv / pip 通常不直接提供通用 `-4` 参数，更适合通过系统地址选择处理：

```text
Windows:
  调整 IPv4/IPv6 前缀策略，或禁用坏网卡 IPv6。

WSL/Ubuntu:
  修改 /etc/gai.conf，让 IPv4 优先。

Clash:
  如果 IPv6 不稳定，设置 ipv6: false，并让 DNS ipv6: false。
```

## 1. 当前机器基线

先把“机器侧配置”和“Clash 侧配置”对齐，后面的所有代理端口都以这里为准。

```text
Windows:
  System Proxy: enabled
  ProxyServer: 127.0.0.1:7897
  WinHTTP proxy: Direct access
  Clash mixed-port: 7897 is listening on 0.0.0.0 / 127.0.0.1 / [::]

WSL:
  distro: Ubuntu-22.04
  WSL mode: mirrored
  autoProxy: false
  dnsTunneling: false
  global proxy env: not set
  git proxy: http://127.0.0.1:7897
  npm proxy: not set
  pip proxy: not set
  /etc/resolv.conf:
    nameserver 198.18.0.2
    nameserver 202.114.96.1
    nameserver 114.114.114.114
```

当前实际端口是 `7897`。如果以后 Clash mixed-port 改回 `7890`，本文所有 `7897` 都要同步改成新的端口。

当前抽样验证：

```text
Windows nslookup:
  DNS server = 198.18.0.2
  github.com / google.com / developer.download.nvidia.com / astral.sh 返回 198.18.x.x fake-ip

Windows curl:
  curl.exe -4 pypi.org / github.com / mirrors.aliyun.com 可通
  curl.exe -6 pypi.org / github.com / mirrors.aliyun.com 解析或连接失败

WSL curl:
  curl -4 pypi.org / github.com / mirrors.aliyun.com 可通
  curl -6 pypi.org / github.com / mirrors.aliyun.com 连接失败
```

因此当前机器应按“IPv4 优先、IPv6 谨慎保留或按网卡禁用”的方向处理。

## 2. 总体原则

这套配置不要把 Windows 和 WSL 当成同一个网络环境处理。

```text
Windows:
  适合使用 Clash System Proxy。
  如果浏览器、ChatGPT、QUIC/UDP 场景需要，可以再开 TUN。

WSL/Ubuntu:
  当前模式下可以先不配置全局代理。
  让 Clash/TUN/DNS 接管默认流量。
  只有某个工具失败时，再对该工具显式使用 127.0.0.1:<mixed-port>。
```

推荐日常稳定组合：

```text
Windows:
  Clash enabled
  System Proxy enabled
  mixed-port = 7897
  IPv6 disabled in Clash if current network IPv6 is unstable
  DNS 优先 redir-host；若使用 fake-ip，必须维护 fake-ip-filter

WSL/Ubuntu:
  networkingMode = mirrored
  autoProxy = false
  默认不设置全局 http_proxy / https_proxy
  git 当前已单独配置 http://127.0.0.1:7897
  其他工具先走默认网络路径
  国内镜像源优先直连真实 IP
  单个工具失败时再显式指定 7897 或 ForceIPv4
```

## 3. Windows 侧配置 Pipeline

### 3.1 确认 Clash 端口

在 Clash Verge Rev 里确认：

```text
Mixed Port: 7897
System Proxy: enabled
TUN Mode: 按需 enabled/disabled
```

PowerShell 验证：

```powershell
netstat -ano | Select-String ':7897'
```

预期：

```text
0.0.0.0:7897 LISTENING
127.0.0.1:7897 available
[::]:7897 LISTENING
```

如果系统代理指向 `127.0.0.1:7897`，但端口没有监听，浏览器、curl、pip、uv 都可能出现间歇失败。

### 3.2 确认 Windows 系统代理

```powershell
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' |
  Select-Object ProxyEnable, ProxyServer, AutoConfigURL
```

当前机器已验证：

```text
ProxyEnable = 1
ProxyServer = 127.0.0.1:7897
```

WinHTTP 单独检查：

```powershell
netsh winhttp show proxy
```

当前机器是 Direct access，这不冲突。浏览器主要看 Windows 用户层系统代理，不看 WinHTTP。

### 3.3 PowerShell 开发工具显式代理

当前 PowerShell 会话临时启用：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
$env:ALL_PROXY="socks5://127.0.0.1:7897"
```

临时关闭：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

验证：

```powershell
curl.exe -I --max-time 10 https://pypi.org/simple/pillow/
curl.exe -I --max-time 10 --proxy http://127.0.0.1:7897 https://pypi.org/simple/pillow/
```

当前机器两条都返回 `HTTP/1.1 200 OK`，说明 Windows 直连/系统代理路径和 Clash mixed-port 都可用。

### 3.4 Clash DNS / IPv6 建议

如果主要目标是开发稳定，优先配置：

```yaml
ipv6: false

dns:
  enable: true
  ipv6: false
  enhanced-mode: redir-host
  nameserver:
    - 223.5.5.5
    - 119.29.29.29
    - 114.114.114.114
  fallback:
    - 1.1.1.1
    - 8.8.8.8
```

如果必须使用 `fake-ip`，国内镜像和开发源至少加入过滤：

```yaml
dns:
  enable: true
  enhanced-mode: fake-ip
  fake-ip-filter:
    - "*.lan"
    - "*.local"
    - "dns.msftncsi.com"
    - "www.msftncsi.com"
    - "time.windows.com"
    - "*.aliyun.com"
    - "*.tsinghua.edu.cn"
    - "*.tuna.tsinghua.edu.cn"
    - "mirrors.tuna.tsinghua.edu.cn"
    - "pypi.tuna.tsinghua.edu.cn"
    - "pypi.org"
    - "*.pypi.org"
    - "files.pythonhosted.org"
    - "*.pythonhosted.org"
    - "github.com"
    - "*.github.com"
    - "*.githubusercontent.com"
```

注意：`fake-ip-filter` 只解决 DNS 返回 fake-ip 的问题，不能保证 WSL mirrored + TUN 的 TCP 连接一定稳定。

### 3.5 Clash 规则建议

规则顺序要放在 `MATCH` 之前：

```yaml
rules:
  - DOMAIN-SUFFIX,pypi.org,Proxy
  - DOMAIN-SUFFIX,pythonhosted.org,Proxy
  - DOMAIN-SUFFIX,github.com,Proxy
  - DOMAIN-SUFFIX,githubusercontent.com,Proxy
  - DOMAIN-SUFFIX,githubassets.com,Proxy

  - DOMAIN-SUFFIX,tuna.tsinghua.edu.cn,DIRECT
  - DOMAIN-SUFFIX,tsinghua.edu.cn,DIRECT
  - DOMAIN-SUFFIX,aliyun.com,DIRECT
  - DOMAIN-SUFFIX,baidu.com,DIRECT
  - DOMAIN-SUFFIX,zhihu.com,DIRECT

  - GEOSITE,CN,DIRECT
  - GEOIP,CN,DIRECT
  - MATCH,Proxy
```

## 4. WSL/Ubuntu 侧配置 Pipeline

### 4.1 确认 WSL 网络模式

Windows 用户目录的 `.wslconfig` 当前为：

```ini
[wsl2]
memory=20GB
swap=30GB

[experimental]
autoMemoryReclaim=gradual
networkingMode=mirrored
dnsTunneling=false
firewall=false
autoProxy=false
```

修改 `.wslconfig` 后需要：

```powershell
wsl --shutdown
```

再重新进入 Ubuntu。

### 4.2 WSL 默认模式：先不设置全局代理

当前验证结果：

```text
WSL env: no http_proxy / https_proxy / all_proxy
git: http.proxy / https.proxy = http://127.0.0.1:7897
npm: proxy / https-proxy = null
pip: no proxy config
```

也就是说，当前并不是所有工具都显式配置了 `7897`。WSL 中直接访问也已验证可用：

```bash
curl -I --max-time 10 https://pypi.org/simple/pillow/
curl -I --max-time 10 https://github.com/
curl -I --max-time 10 https://mirrors.tuna.tsinghua.edu.cn/
```

当前机器三条都返回 `HTTP/2 200`。

所以日常默认策略应简化为：

```text
不要默认 export 全局代理变量。
git 保留单独代理配置。
pip / npm / curl / apt 先走默认网络路径。
只有某个工具失败时，再临时走 127.0.0.1:7897。
```

### 4.3 WSL 代理函数：应急开关

写入 `~/.bashrc` 或 `~/.zshrc`：

```bash
# WSL Ubuntu -> Windows Clash
_PROXY_HOST="127.0.0.1"
_PROXY_PORT="7897"
_PROXY_HTTP="http://${_PROXY_HOST}:${_PROXY_PORT}"
_PROXY_SOCKS="socks5://${_PROXY_HOST}:${_PROXY_PORT}"
_NO_PROXY="localhost,127.0.0.1,::1,.local,10.*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,192.168.*"

proxy_on() {
  export http_proxy="${_PROXY_HTTP}"
  export https_proxy="${_PROXY_HTTP}"
  export all_proxy="${_PROXY_SOCKS}"
  export HTTP_PROXY="${_PROXY_HTTP}"
  export HTTPS_PROXY="${_PROXY_HTTP}"
  export ALL_PROXY="${_PROXY_SOCKS}"
  export no_proxy="${_NO_PROXY}"
  export NO_PROXY="${_NO_PROXY}"
  echo "Proxy ON -> ${_PROXY_HTTP}"
}

proxy_off() {
  unset http_proxy https_proxy all_proxy no_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  echo "Proxy OFF"
}

proxy_status() {
  if [ -n "${http_proxy}" ]; then
    echo "Proxy ON -> ${http_proxy}"
  else
    echo "Proxy OFF"
  fi
}

proxy_run() {
  http_proxy="${_PROXY_HTTP}" \
  https_proxy="${_PROXY_HTTP}" \
  all_proxy="${_PROXY_SOCKS}" \
  HTTP_PROXY="${_PROXY_HTTP}" \
  HTTPS_PROXY="${_PROXY_HTTP}" \
  ALL_PROXY="${_PROXY_SOCKS}" \
  "$@"
}

proxysudo() {
  sudo -E env \
  http_proxy="${_PROXY_HTTP}" \
  https_proxy="${_PROXY_HTTP}" \
  all_proxy="${_PROXY_SOCKS}" \
  "$@"
}
```

这段函数建议保留，但不要默认执行 `proxy_on`。需要单次代理时用：

```bash
proxy_run curl -I https://pypi.org/simple/pillow/
proxysudo apt-get update
```

### 4.4 WSL 代理连通性验证

```bash
curl -I --max-time 10 --proxy http://127.0.0.1:7897 https://pypi.org/simple/pillow/
```

当前机器已验证返回：

```text
HTTP/1.1 200 Connection established
HTTP/2 200
```

这说明在 mirrored 模式下，WSL 通过 `127.0.0.1:7897` 可以访问 Windows Clash。

如果以后 `127.0.0.1` 不通，再退回 Windows 主机 IP：

```bash
grep nameserver /etc/resolv.conf
export http_proxy=http://<windows-host-ip>:7897
export https_proxy=http://<windows-host-ip>:7897
```

### 4.5 WSL DNS 检查

```bash
cat /etc/resolv.conf
getent ahosts pypi.org | head
getent ahosts mirrors.tuna.tsinghua.edu.cn | head
```

当前机器已验证：

```text
pypi.org -> 198.18.0.21
mirrors.tuna.tsinghua.edu.cn -> 101.6.15.130
```

解释：

```text
pypi.org 返回 198.18.x.x:
  Clash fake-ip/DNS 正在影响 WSL。
  这不是必然错误；当前直接 curl 仍可通，说明默认接管链路可用。
  只有连接失败时，才需要改为显式走 Clash 端口。

清华镜像返回真实 IP:
  国内镜像 fake-ip-filter / DNS 分流基本符合预期。
```

### 4.6 WSL DNS 崩坏时的修复

只有出现 `Temporary failure resolving`、`apt update` 解析失败、`resolv.conf` 被异常 IPv6 DNS 污染时，再固定 DNS。

```bash
sudo nano /etc/wsl.conf
```

写入：

```ini
[network]
generateResolvConf = false
```

重建 `/etc/resolv.conf`：

```bash
sudo rm -f /etc/resolv.conf
sudo bash -c 'echo "nameserver 114.114.114.114" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 223.5.5.5" >> /etc/resolv.conf'
sudo bash -c 'echo "nameserver 119.29.29.29" >> /etc/resolv.conf'
```

如果仍被覆盖，再使用：

```bash
sudo chattr +i /etc/resolv.conf
```

恢复自动 DNS 前必须先解除：

```bash
sudo chattr -i /etc/resolv.conf
```

### 4.7 WSL/Ubuntu IPv4 优先

如果 `curl -4` 正常、`curl -6` 失败，或 apt/pip/uv 经常卡在 IPv6 路径，优先调整地址选择，而不是直接全局禁 IPv6。

```bash
sudo nano /etc/gai.conf
```

取消注释或添加：

```conf
precedence ::ffff:0:0/96  100
```

然后重启 WSL：

```powershell
wsl --shutdown
```

### 4.8 apt 使用方式

默认先直接使用：

```bash
sudo apt-get update
sudo apt-get install git
```

如果默认路径失败，再临时代理：

```bash
proxysudo apt-get update
proxysudo apt-get install git
```

测试 IPv4：

```bash
sudo apt-get -o Acquire::ForceIPv4=true update
```

如果 ForceIPv4 明显更稳定，再永久配置：

```bash
echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4
```

不建议默认写死 apt 代理：

```conf
Acquire::http::Proxy "http://127.0.0.1:7897";
Acquire::https::Proxy "http://127.0.0.1:7897";
```

因为 Clash 没启动时 apt 会直接失败。更稳的方式是默认直连，失败时再用 `proxysudo`。

## 5. 网络体检 Pipeline

目标不是只证明某个网站能打开，而是判断这台机器是否适合日常开发、浏览、下载、安装依赖。测试顺序应该从底层到工具层：

```text
网卡 IPv4/IPv6
  ↓
DNS 解析
  ↓
Clash 端口 / 系统代理 / TUN 接管
  ↓
浏览与下载站点
  ↓
WSL/Ubuntu 开发工具链
  ↓
失败时再切换到显式代理或 ForceIPv4
```

### 5.1 Windows 基础体检

```powershell
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed

Get-NetAdapterBinding -Name "*" | Where-Object {
  $_.ComponentID -in "ms_tcpip", "ms_tcpip6"
} | Format-Table Name, DisplayName, ComponentID, Enabled

Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' |
  Select-Object ProxyEnable, ProxyServer, AutoConfigURL

netsh winhttp show proxy
netstat -ano | Select-String ':7897'
```

预期：

```text
当前联网网卡 IPv4 启用。
IPv6 如果启用，必须通过后面的 curl -6 证明可用。
Windows System Proxy 指向 Clash mixed-port。
Clash mixed-port 正在监听。
```

### 5.2 Windows DNS 与 IPv4/IPv6 体检

```powershell
$hosts = @(
  "pypi.org",
  "files.pythonhosted.org",
  "mirrors.aliyun.com",
  "pypi.tuna.tsinghua.edu.cn",
  "mirrors.tuna.tsinghua.edu.cn",
  "github.com",
  "raw.githubusercontent.com",
  "objects.githubusercontent.com",
  "google.com",
  "developer.download.nvidia.com",
  "astral.sh",
  "nodejs.org",
  "registry.npmjs.org",
  "docker.io",
  "ghcr.io"
)

foreach ($h in $hosts) {
  Write-Host "`n=== nslookup $h ==="
  nslookup $h
}

$urls = @(
  "https://pypi.org/simple/pillow/",
  "https://files.pythonhosted.org/",
  "https://mirrors.aliyun.com/pypi/simple/",
  "https://pypi.tuna.tsinghua.edu.cn/simple/",
  "https://mirrors.tuna.tsinghua.edu.cn/",
  "https://github.com/",
  "https://raw.githubusercontent.com/",
  "https://objects.githubusercontent.com/",
  "https://google.com/",
  "https://developer.download.nvidia.com/compute/cuda/repos/",
  "https://astral.sh/",
  "https://nodejs.org/dist/",
  "https://registry.npmjs.org/",
  "https://hub.docker.com/",
  "https://ghcr.io/"
)

foreach ($u in $urls) {
  Write-Host "`n=== IPv4 $u ==="
  curl.exe -4 -I --max-time 10 $u
  Write-Host "`n=== IPv6 $u ==="
  curl.exe -6 -I --max-time 10 $u
}
```

判断：

```text
IPv4 大多数成功，IPv6 大多数失败：
  当前机器按 IPv4 优先处理。

国内镜像返回 198.18.x.x：
  检查 fake-ip-filter，国内镜像更适合真实 IP + DIRECT。

GitHub / Google / NVIDIA / registry 等返回 198.18.x.x：
  说明 Clash fake-ip 正在接管，这本身不是错误。

curl -4 成功但默认 curl 失败：
  系统或工具可能优先走了坏 IPv6。
```

### 5.3 Windows 浏览与下载体检

```powershell
curl.exe -I --max-time 10 https://www.microsoft.com/
curl.exe -I --max-time 10 https://www.baidu.com/
curl.exe -I --max-time 10 https://www.zhihu.com/
curl.exe -I --max-time 10 https://www.bing.com/
curl.exe -I --max-time 10 https://chatgpt.com/
curl.exe -I --max-time 10 https://github.com/
curl.exe -I --max-time 10 https://developer.download.nvidia.com/compute/cuda/repos/
curl.exe -I --max-time 10 --proxy http://127.0.0.1:7897 https://github.com/
```

预期：

```text
国内站点直连或 DIRECT 正常。
国外站点能被 Clash 接管。
显式 --proxy 127.0.0.1:7897 可作为对照路径。
```

### 5.4 WSL/Ubuntu 基础体检

```bash
echo "=== WSL release ==="
cat /etc/os-release

echo "=== proxy env ==="
env | grep -i '_proxy' || true

echo "=== resolv.conf ==="
cat /etc/resolv.conf

echo "=== default routes ==="
ip route
ip -6 route || true

echo "=== git proxy ==="
git config --global --get-regexp 'http.*proxy|https.*proxy' || true
```

预期：

```text
默认可以没有全局 http_proxy / https_proxy。
resolv.conf 不应是完全不可用的 DNS。
git 如果单独配置 7897，要确认端口与 Clash mixed-port 一致。
```

### 5.5 WSL/Ubuntu DNS 与 IPv4/IPv6 体检

```bash
for h in \
  pypi.org \
  files.pythonhosted.org \
  mirrors.aliyun.com \
  pypi.tuna.tsinghua.edu.cn \
  mirrors.tuna.tsinghua.edu.cn \
  github.com \
  raw.githubusercontent.com \
  objects.githubusercontent.com \
  google.com \
  developer.download.nvidia.com \
  astral.sh \
  nodejs.org \
  registry.npmjs.org \
  docker.io \
  ghcr.io
do
  echo "=== getent $h ==="
  getent ahosts "$h" | head
done

for u in \
  https://pypi.org/simple/pillow/ \
  https://files.pythonhosted.org/ \
  https://mirrors.aliyun.com/pypi/simple/ \
  https://pypi.tuna.tsinghua.edu.cn/simple/ \
  https://mirrors.tuna.tsinghua.edu.cn/ \
  https://github.com/ \
  https://raw.githubusercontent.com/ \
  https://objects.githubusercontent.com/ \
  https://google.com/ \
  https://developer.download.nvidia.com/compute/cuda/repos/ \
  https://astral.sh/ \
  https://nodejs.org/dist/ \
  https://registry.npmjs.org/ \
  https://hub.docker.com/ \
  https://ghcr.io/
do
  echo "=== IPv4 $u ==="
  curl -4 -I --max-time 10 "$u"
  echo "=== IPv6 $u ==="
  curl -6 -I --max-time 10 "$u"
done
```

判断和 Windows 一样：先定 IPv4/IPv6，再谈代理。WSL 下如果 `curl -6` 普遍失败，就把 `/etc/gai.conf` 的 IPv4 优先作为默认处理。

### 5.6 WSL/Ubuntu 工具链体检

这组命令不以安装东西为目标，尽量使用只读或轻量请求。

```bash
echo "=== curl direct ==="
curl -I --max-time 10 https://github.com/
curl -I --max-time 10 https://pypi.org/simple/pillow/
curl -I --max-time 10 https://mirrors.aliyun.com/pypi/simple/

echo "=== curl via Clash port ==="
curl -I --max-time 10 --proxy http://127.0.0.1:7897 https://github.com/
curl -I --max-time 10 --proxy http://127.0.0.1:7897 https://pypi.org/simple/pillow/

echo "=== git ==="
git ls-remote https://github.com/git/git.git HEAD

echo "=== apt ==="
sudo apt-get update
sudo apt-get -o Acquire::ForceIPv4=true update

echo "=== pip ==="
python3 -m pip config list -v 2>/dev/null || true
python3 -m pip index versions pip --timeout 10

echo "=== uv ==="
command -v uv >/dev/null 2>&1 && uv --version || true
command -v uv >/dev/null 2>&1 && uv pip install --dry-run --python "$(command -v python3)" --index-url https://pypi.org/simple pip || true

echo "=== npm ==="
command -v npm >/dev/null 2>&1 && npm config get proxy || true
command -v npm >/dev/null 2>&1 && npm config get https-proxy || true
command -v npm >/dev/null 2>&1 && npm view npm version --registry=https://registry.npmjs.org/ || true

echo "=== node / cargo / go optional ==="
command -v node >/dev/null 2>&1 && node --version || true
command -v cargo >/dev/null 2>&1 && cargo search serde --limit 1 || true
command -v go >/dev/null 2>&1 && GOPROXY=https://proxy.golang.org,direct go env GOPROXY || true
```

判断：

```text
curl direct 成功：
  WSL 默认接管路径可用。

curl direct 失败，但 curl --proxy 7897 成功：
  该工具可以临时显式代理，不必重做整套网络。

git 失败：
  先看 git proxy 是否和 Clash mixed-port 一致，再测 GitHub 规则。

apt 普通 update 失败但 ForceIPv4 成功：
  apt 走了坏 IPv6，配置 Acquire::ForceIPv4。

pip / uv / npm 失败：
  先看是否 DNS/IPv6 问题，再看是否需要临时设置代理或换源。
```

### 5.7 体检通过标准

一台“开发上基本好用”的机器，至少满足：

```text
Windows:
  当前联网网卡 IPv4 正常。
  如果 IPv6 开启，curl -6 不能大面积失败；否则应降低 IPv6 优先级或禁用坏网卡 IPv6。
  Clash mixed-port 与 Windows System Proxy 一致。
  浏览、GitHub、Google、NVIDIA、PyPI、国内镜像源可访问。

WSL/Ubuntu:
  DNS 能解析国内镜像和国外开发站点。
  curl direct 能访问 PyPI / GitHub / 阿里源 / 清华源。
  curl --proxy 127.0.0.1:7897 可作为兜底通道。
  git ls-remote GitHub 成功。
  apt update 或 apt ForceIPv4 update 至少一条成功。
  pip / uv / npm 的查询类命令能成功。
```

如果这些都通过，日常开发、浏览、依赖安装、源码拉取、模型/驱动下载大概率没有大问题。

## 6. 排查顺序

遇到网络问题时按这个顺序，不要同时改很多层：

1. 先用 `nslookup`、`curl -4`、`curl -6` 判断当前机器 IPv4/IPv6 哪条链路可靠。
2. 如果 IPv6 不可靠，先调整 Windows/WSL 的 IPv4 优先级；确认是某张网卡坏 IPv6 时，再禁用那张网卡的 IPv6。
3. 确认 Clash mixed-port 是多少。
4. 确认 Windows 系统代理端口是否等于 mixed-port。
5. `curl.exe --proxy http://127.0.0.1:<port>` 测 Windows Clash 端口。
6. WSL 里先不用代理直接 `curl` PyPI / GitHub / Google / 阿里源 / 清华源 / NVIDIA / uv 相关站点。
7. 如果直接失败，再用 `curl --proxy http://127.0.0.1:<port>` 测 WSL 到 Windows Clash。
8. 检查 WSL `/etc/resolv.conf` 和 `getent ahosts`。
9. 国内源失败时检查是否返回 fake-ip，以及 Clash 是否把国内源 DIRECT。
10. 国外源失败时再对该工具显式走代理。
11. DNS 解析失败才固定 `/etc/resolv.conf`。
12. TUN 打开后才出问题，就先回到 System Proxy；必要时对失败工具显式代理。

## 7. 一句话结论

当前机器最匹配的稳定方案是：

```text
先判断机器 IPv6 是否可靠；不可靠时优先 IPv4，必要时禁用坏网卡 IPv6；
Windows 使用 Clash System Proxy，端口 7897；
WSL mirrored 当前不需要给所有开发工具显式配置 7897；
git 保留单独代理，其余工具优先使用默认接管路径；
国内镜像源尽量真实 IP + DIRECT；
国外开发源当前可由 Clash 默认接管，失败时再显式代理；
网络测试要覆盖 PyPI / files.pythonhosted / 阿里源 / 清华源 / GitHub / Google / NVIDIA / astral.sh。
```
