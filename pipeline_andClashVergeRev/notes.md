
---

# 一、先讲清楚整体逻辑

## 1. 你这个问题的核心矛盾

你的环境大概是：

```text
Windows 主机
  └─ Clash Verge Rev / mihomo
      └─ 开启 TUN 模式

WSL2
  └─ networkingMode = mirrored
  └─ 主要运行 apt / pip / git / curl / npm 等开发工具

当前机器
  └─ IPv6 不可用或不稳定
```

你 PDF 里已经明确记录了一个关键结论：**Windows 侧继续用 TUN 是有价值的，因为 ChatGPT、浏览器、QUIC/UDP 等流量更稳定；但 WSL2 侧更适合显式走 7890 代理端口，绕开 TUN 对 mirrored 网络的干扰。** PDF 第 26 页把你的环境总结为“Windows + Clash TUN、WSL2 mirrored、无 IPv6 网络环境”，并给出策略：Windows 继续 TUN，WSL2 全部走 7890 端口。

这个判断是合理的。

更精确地说：

```text
Windows 本机：
    TUN 很适合，因为可以接管 TCP / UDP / QUIC。

WSL2 开发环境：
    apt / pip / git / curl 主要是 TCP。
    显式 http_proxy / https_proxy 走 7890 更简单、更稳定。
```

---

## 2. 普通代理模式和 TUN 模式不是一回事

你的 PDF 第 21-23 页对这部分讲得比较完整。普通代理模式是：

```text
应用程序
  ↓ 主动设置 http_proxy / https_proxy / socks_proxy
Clash 7890 端口
  ↓
Clash 按规则 DIRECT / PROXY / REJECT
```

也就是说，程序必须主动知道代理存在。

而 TUN 模式是：

```text
应用程序
  ↓ 不需要设置代理
操作系统路由表
  ↓
TUN 虚拟网卡
  ↓
Clash 接管 IP 层流量
```

PDF 第 22 页明确写到：代理模式只处理主动设置代理的流量，而 TUN 模式是在网卡层处理，劫持所有流量，包括 TCP 和 UDP。

所以两者的本质区别是：

```text
代理模式：应用层代理
TUN 模式：网络层劫持
```

这也是为什么你会看到两类完全不同的现象：

```text
浏览器 / ChatGPT：
    可能依赖 QUIC / UDP / WebSocket
    TUN 更稳定

apt / pip / git / curl：
    基本是 TCP
    7890 代理端口已经够用
```

---

## 3. 为什么 ChatGPT 这类应用 TUN 更稳定

PDF 第 23 页指出了关键原因：**UDP 和 DNS 处理方式不同。**
普通 HTTP 代理通常主要处理 TCP，不能完整接管 UDP/QUIC。ChatGPT、浏览器、部分现代网站可能使用 QUIC，也可能在 WebSocket、HTTP/2、HTTP/3 之间切换。

所以普通代理模式下可能是：

```text
TCP → 走代理
UDP / QUIC → 没走代理，直接从真实网卡出去
```

这就会导致：

```text
能打开
但不稳定

能登录
但中途断流

看起来代理可用
但 ChatGPT 一会儿好一会儿坏
```

TUN 模式的优势是：

```text
TCP → 接管
UDP → 接管
DNS → 接管
QUIC → 接管
```

所以 Windows 侧继续用 TUN 是有道理的。

---

## 4. 为什么 WSL2 mirrored + TUN 反而容易出问题

这部分是你的笔记里最重要、也最容易误解的地方。

WSL2 mirrored 模式会让 WSL2 更像直接共享 Windows 的网络环境，但它不是简单地“WSL 等于 Windows 本机”。在 mirrored 模式下，WSL2 应用发出的流量要经过 WSL2/Windows 共享网络栈，再可能被 Windows 侧 Clash TUN 虚拟网卡接管。

你的 PDF 第 30-31 页对这个问题做了一个修正：**不应简单说“TUN 无故干扰 TCP”，更准确地说，是 WSL2 mirrored 模式下，流量被 Windows TUN 接管后，源地址、连接上下文、接管时序可能出现兼容性问题，从而导致连接 reset。**

也就是说，问题不是：

```text
Clash 规则错了
```

也不只是：

```text
fake-ip 错了
```

而更像是：

```text
WSL2 应用发起 TCP 连接
  ↓
mirrored 网络栈转发到 Windows 网络层
  ↓
Windows TUN 虚拟网卡接管
  ↓
Clash 决定 DIRECT 或 PROXY
  ↓
重新发起真实连接
  ↓
上下文/源地址/接管时机和 Windows 本机流量不完全一致
  ↓
连接被 reset、卡住、握手异常
```

因此你遇到的“国内镜像源加了 fake-ip-filter 但还是连接不上”，并不矛盾。因为：

```text
fake-ip-filter 只解决 DNS 层；
连接 reset 发生在 TCP 流量层。
```

PDF 第 21 页也写到了这个现象：fake-ip-filter 解决的是 DNS 层，但流量层仍然可能被 TUN 拦截，导致连接 reset。

---

## 5. fake-ip 到底是什么，为什么会看到 198.18.x.x

你的 PDF 第 24-30 页对 fake-ip 机制整理得比较完整。简单说：

```text
198.18.x.x 不是目标网站的真实 IP
它是 Clash 分配给某个域名的“虚拟标签”
```

例如：

```text
google.com
  ↓ DNS 查询
Clash DNS 返回 198.18.0.22
  ↓
应用连接 198.18.0.22
  ↓
TUN 拦截
  ↓
Clash 查映射表：
    198.18.0.22 = google.com
  ↓
按规则走代理节点
```

PDF 第 29-30 页也明确写到：198.18.0.x 本身不是服务器，它只是一个标签，真正的中转由 Clash 和代理节点完成。

所以你之前记得的 `192.18 xxx`，大概率其实是：

```text
198.18.x.x
```

这是 Clash fake-ip 的典型网段。

问题在于：

```text
fake-ip 正常工作的前提是：
DNS 查询和后续 TCP/UDP 连接都被同一个 Clash 实例正确接管。
```

如果 WSL2 里拿到了 fake-ip，但后续连接没被正确接管，就会变成：

```text
应用连接 198.18.x.x
  ↓
但 Clash 没有正确完成映射/转发
  ↓
卡住、reset、timeout
```

---

## 6. fake-ip-filter 能解决什么，不能解决什么

你的 PDF 第 1 页、第 17-18 页都强调了 fake-ip-filter。配置里把这些国内镜像加入 filter：

```yaml
fake-ip-filter:
  - '*.aliyun.com'
  - '*.tsinghua.edu.cn'
  - '*.mirrors.ustc.edu.cn'
  - 'mirrors.tuna.tsinghua.edu.cn'
  - '*.npmjs.org'
  - '*.pypi.org'
```

作用是：

```text
这些域名不要返回 198.18.x.x
而是返回真实 IP
```

PDF 第 18 页也给了验证方法：国内镜像应该返回真实 IP，google/github 这类国外域名可以返回 198.18.x.x。

但是注意：

```text
fake-ip-filter 只影响 DNS 返回结果。
```

它不能保证：

```text
后续 TCP 连接一定不会被 TUN 接管；
后续 HTTPS 握手一定不会 reset；
WSL2 mirrored 和 Windows TUN 一定兼容。
```

所以它是必要配置，但不一定是充分条件。

---

## 7. 为什么 IPv6 会让另一台机器“看起来没问题”

PDF 第 31-33 页提到一个很重要的现象：你的 Clash TUN 配置里只有：

```json
"inet4-address": ["198.18.0.1/30"]
```

没有：

```json
"inet6-address"
```

这意味着：

```text
TUN 主要接管 IPv4；
IPv6 流量可能没有被 TUN 接管。
```

PDF 第 32 页明确写到：如果只有 `inet4-address`，没有 `inet6-address`，IPv6 流量不经过 TUN，会直接走真实网卡。

所以会出现这种现象：

```text
机器 A 有 IPv6：
    访问清华源 / 阿里源时可能直接走 IPv6
    绕过了 TUN 的 IPv4 接管问题
    所以看起来正常

机器 B 没有 IPv6：
    只能走 IPv4
    IPv4 被 TUN 接管
    WSL2 mirrored + TUN 兼容性问题暴露
    所以访问失败或卡住
```

这解释了你说的“机器只支持 IPv4，需要改某个配置”的背景。

不过这里要区分两个方向：

```text
方向 1：让 WSL 优先 IPv4
    适合 IPv6 不稳定、经常卡住的情况。

方向 2：手写 resolv.conf 为 IPv4 DNS
    适合 WSL DNS 解析本身坏掉的情况。
```

它们不是一回事。

---

## 8. apt update 卡住或 Temporary failure resolving 的根因

PDF 第 33-34 页记录了这个问题：

```text
sudo apt update 出现：
Temporary failure resolving 'mirrors.tuna.tsinghua.edu.cn'
Failed to fetch http://mirrors.tuna.tsinghua.edu.cn/...
```

这说明至少在那个时刻，问题发生在 DNS 解析阶段。

可能链路是：

```text
apt update
  ↓
解析 mirrors.tuna.tsinghua.edu.cn
  ↓
WSL resolv.conf 指向异常 DNS
  ↓
DNS 无法解析
  ↓
Temporary failure resolving
```

PDF 第 2 页也记录了一个典型异常 resolv.conf：

```text
nameserver fec0:0:0:ffff::1
nameserver fec0:0:0:ffff::2
nameserver fec0:0:0:ffff::3
```

这其实是 IPv6 形式的 DNS 地址。如果当前机器 IPv6 不可用、不稳定，或者 WSL DNS tunneling/Windows DNS 链路有问题，apt 就会解析失败。

因此你的修复思路是：

```text
禁止 WSL 自动生成 resolv.conf
手写 IPv4 DNS
```

也就是 PDF 第 33-34 页的做法：

```bash
sudo nano /etc/wsl.conf

[network]
generateResolvConf = false

sudo rm /etc/resolv.conf
sudo bash -c 'echo "nameserver 114.114.114.114" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 8.8.8.8" >> /etc/resolv.conf'
sudo chattr +i /etc/resolv.conf
```

这个方法确实能“强行稳定 DNS”，但我建议把它定位为：

```text
DNS 崩坏时的强制修复方案
```

而不是所有机器的默认最优方案。因为 `chattr +i /etc/resolv.conf` 会让这个文件不可修改，以后如果你想恢复 WSL 自动 DNS、dnsTunneling 或 VPN DNS，会被这个不可变属性卡住。

恢复时要执行：

```bash
sudo chattr -i /etc/resolv.conf
```

---

# 二、整理后的新笔记

下面是我建议你保留的新版本。

---

# WSL2 mirrored + Clash TUN 网络问题整理笔记

## 1. 问题背景

当前环境：

```text
Windows:
  Clash Verge Rev / mihomo
  开启 TUN 模式

WSL2:
  networkingMode = mirrored
  主要使用 apt / pip / git / curl / npm 等开发工具

网络特征:
  当前机器 IPv6 不可用或不稳定
  WSL2 中 apt update 可能卡住或 DNS 解析失败
```

典型故障：

```text
1. Windows 侧可以访问，WSL2 侧访问失败。
2. WSL2 中 apt update 卡在 headers。
3. apt update 报 Temporary failure resolving。
4. 国内镜像源，如 mirrors.tuna.tsinghua.edu.cn，连接失败。
5. DNS 查询返回 198.18.x.x。
6. 加了 fake-ip-filter 后，DNS 看起来对了，但 HTTPS 仍然 reset。
```

---

## 2. 普通代理模式和 TUN 模式的区别

### 2.1 普通代理模式

```text
应用程序
  ↓ 主动设置 http_proxy / https_proxy / all_proxy
Clash 7890 端口
  ↓
Clash 规则引擎
  ↓
DIRECT / PROXY / REJECT
```

特点：

```text
优点：
  配置简单
  对 apt / pip / git / curl 这类 TCP 工具足够
  在 WSL2 中更稳定

缺点：
  需要程序支持代理
  UDP / QUIC 可能不走代理
  ping / traceroute / 某些系统流量不会走代理
```

---

### 2.2 TUN 模式

```text
应用程序
  ↓ 不需要设置代理
操作系统路由表
  ↓
TUN 虚拟网卡
  ↓
Clash 接管 TCP / UDP / DNS
  ↓
DIRECT / PROXY / REJECT
```

特点：

```text
优点：
  程序无感知
  TCP / UDP / QUIC 都能接管
  对 ChatGPT、浏览器、游戏、视频通话等更完整

缺点：
  更依赖系统路由、虚拟网卡、DNS 劫持
  和 WSL2 mirrored、VPN、虚拟网卡容易冲突
  配置复杂度更高
```

---

## 3. 推荐模式选择

判断标准：

```text
如果应用主要是：
  apt / pip / git / curl / wget / npm
  → 普通代理 7890 足够

如果应用依赖：
  UDP / QUIC / WebSocket / 游戏 / 实时通信 / ChatGPT 网页
  → TUN 更合适
```

针对当前环境，推荐：

```text
Windows 本机：
  继续使用 Clash TUN
  保障 ChatGPT、浏览器、QUIC/UDP 场景稳定

WSL2：
  不强依赖 TUN
  显式走 Clash 7890 HTTP/SOCKS 代理
  避免 WSL2 mirrored + Windows TUN 的兼容性问题
```

---

## 4. fake-ip 机制

### 4.1 fake-ip 正常流程

```text
应用查询 google.com
  ↓
Clash DNS 返回 198.18.0.22
  ↓
Clash 建立映射：
  198.18.0.22 ↔ google.com
  ↓
应用连接 198.18.0.22
  ↓
TUN 拦截
  ↓
Clash 查映射表，还原 google.com
  ↓
匹配规则
  ↓
走代理节点
```

注意：

```text
198.18.x.x 不是真实服务器。
它只是 Clash fake-ip 模式下的虚拟标签。
```

---

### 4.2 fake-ip-filter 的作用

对于国内镜像源，不希望返回 fake-ip，而希望返回真实 IP：

```yaml
dns:
  enable: true
  enhanced-mode: fake-ip
  fake-ip-filter:
    - '*.aliyun.com'
    - '*.tsinghua.edu.cn'
    - '*.mirrors.ustc.edu.cn'
    - 'mirrors.tuna.tsinghua.edu.cn'
    - '*.npmjs.org'
    - '*.pypi.org'
```

效果：

```text
mirrors.tuna.tsinghua.edu.cn
  → 返回真实 IP

google.com / github.com
  → 可以返回 198.18.x.x fake-ip
```

验证：

```bash
dig mirrors.aliyun.com @127.0.0.1 -p 1053
dig mirrors.tuna.tsinghua.edu.cn @127.0.0.1 -p 1053
dig pypi.tuna.tsinghua.edu.cn @127.0.0.1 -p 1053

dig google.com @127.0.0.1 -p 1053
dig github.com @127.0.0.1 -p 1053
```

预期：

```text
国内镜像：
  返回真实 IP

国外域名：
  返回 198.18.x.x
```

---

## 5. fake-ip-filter 解决不了所有问题

必须区分两层：

```text
DNS 层：
  fake-ip-filter 决定返回真实 IP 还是 fake-ip

流量层：
  应用拿到 IP 后，继续发起 TCP / UDP 连接
```

因此：

```text
fake-ip-filter 生效
≠
HTTPS 一定不会 reset

DNS 返回真实 IP
≠
TUN 不会接管这条连接

Clash 规则 DIRECT
≠
WSL2 mirrored + TUN 一定兼容
```

如果 WSL2 中出现：

```text
DNS 已经返回真实 IP
但 curl / apt / pip 仍然 reset 或 timeout
```

更可能是：

```text
WSL2 mirrored + Windows TUN 的流量接管时序/上下文问题
```

此时不要继续只改 fake-ip-filter，而应该让 WSL2 显式走 7890 端口，绕开 TUN 接管。

---

## 6. IPv4 / IPv6 问题

### 6.1 当前 TUN 只接管 IPv4 的情况

如果 Clash API 看到：

```json
{
  "inet4-address": ["198.18.0.1/30"]
}
```

但没有：

```json
"inet6-address"
```

说明：

```text
IPv4 流量被 TUN 接管
IPv6 流量可能绕过 TUN
```

这会导致：

```text
有 IPv6 的机器：
  国内镜像可能走 IPv6 直连
  绕过 TUN 问题
  所以正常

无 IPv6 的机器：
  只能走 IPv4
  IPv4 被 TUN 接管
  WSL2 mirrored + TUN 问题暴露
```

---

### 6.2 让 WSL 优先 IPv4

如果 IPv6 不稳定，可以设置 glibc 地址选择策略：

```bash
sudo nano /etc/gai.conf
```

取消注释：

```conf
precedence ::ffff:0:0/96  100
```

作用：

```text
当域名同时有 IPv4 和 IPv6 时，优先使用 IPv4。
```

验证：

```bash
getent ahosts mirrors.tuna.tsinghua.edu.cn
getent ahosts github.com
```

---

## 7. WSL2 DNS 修复

### 7.1 DNS 问题的典型表现

```text
sudo apt update
```

报错：

```text
Temporary failure resolving 'mirrors.tuna.tsinghua.edu.cn'
Failed to fetch http://mirrors.tuna.tsinghua.edu.cn/...
```

说明 apt 连第一步 DNS 解析都没完成。

检查：

```bash
cat /etc/resolv.conf
```

如果看到：

```text
nameserver fec0:0:0:ffff::1
nameserver fec0:0:0:ffff::2
nameserver fec0:0:0:ffff::3
```

而当前机器 IPv6 不可用，则可能导致 DNS 失败。

---

### 7.2 强制使用 IPv4 DNS

编辑：

```bash
sudo nano /etc/wsl.conf
```

写入：

```ini
[network]
generateResolvConf = false
```

删除旧 resolv.conf：

```bash
sudo rm -f /etc/resolv.conf
```

写入可靠 IPv4 DNS：

```bash
sudo bash -c 'echo "nameserver 114.114.114.114" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 223.5.5.5" >> /etc/resolv.conf'
sudo bash -c 'echo "nameserver 119.29.29.29" >> /etc/resolv.conf'
```

如果仍然被 WSL 自动覆盖，可以加不可变属性：

```bash
sudo chattr +i /etc/resolv.conf
```

重启 WSL：

```powershell
wsl --shutdown
```

重新进入 WSL 后验证：

```bash
cat /etc/resolv.conf
getent hosts mirrors.tuna.tsinghua.edu.cn
```

注意：
如果以后要恢复自动 DNS，需要先执行：

```bash
sudo chattr -i /etc/resolv.conf
```

---

## 8. WSL2 推荐代理配置

建议在 `~/.zshrc` 或 `~/.bashrc` 里使用函数，不要完全不可控地污染环境。

### 8.1 统一代理变量

```bash
# ================================================================
# Proxy Configuration for WSL2 + Clash
# ================================================================

_PROXY_HOST="127.0.0.1"
_PROXY_PORT="7890"
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
  if [[ -n "${http_proxy}" ]]; then
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

建议：

```text
不要盲目默认 proxy_on。
更推荐需要时手动 proxy_on，或者用 proxy_run 单次代理。
```

如果你确认 WSL2 开发环境长期都需要代理，再在最后加：

```bash
proxy_on
```

---

## 9. apt 的推荐使用方式

### 9.1 临时走代理

```bash
proxysudo apt-get update
proxysudo apt-get install git
```

或者：

```bash
proxy_run sudo -E apt-get update
```

### 9.2 强制 IPv4 测试

```bash
sudo apt-get -o Acquire::ForceIPv4=true update
```

如果这条能成功，而普通 `apt update` 不行，说明 IPv6 相关链路有问题。

可永久配置：

```bash
echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4
```

### 9.3 不建议一开始就写死 apt 代理

不优先建议直接写：

```bash
sudo nano /etc/apt/apt.conf.d/99proxy
```

除非你确定 apt 永远要走代理。

因为一旦写死：

```conf
Acquire::http::Proxy "http://127.0.0.1:7890";
Acquire::https::Proxy "http://127.0.0.1:7890";
```

以后 Clash 没启动时，apt 会直接失败。

更推荐：

```text
日常：
  apt 直连国内源

需要代理时：
  proxysudo apt-get update
```

---

## 10. Clash Verge Rev 推荐配置

### 10.1 DNS 覆写

```yaml
dns:
  enable: true
  enhanced-mode: fake-ip
  fake-ip-filter:
    - '*.aliyun.com'
    - '*.tsinghua.edu.cn'
    - '*.mirrors.ustc.edu.cn'
    - 'mirrors.tuna.tsinghua.edu.cn'
    - 'pypi.tuna.tsinghua.edu.cn'
    - '*.npmjs.org'
    - '*.pypi.org'
    - 'registry.npmmirror.com'
    - 'hf-mirror.com'
```

如果机器 IPv6 不可用：

```yaml
dns:
  ipv6: false
```

---

### 10.2 规则补充

```yaml
rules:
  - PROCESS-NAME,clash-verge*,DIRECT
  - DST-PORT,7890,DIRECT
  - DST-PORT,7897,DIRECT

  - DOMAIN-SUFFIX,aliyun.com,DIRECT
  - DOMAIN-SUFFIX,tsinghua.edu.cn,DIRECT
  - DOMAIN-SUFFIX,tuna.tsinghua.edu.cn,DIRECT
  - DOMAIN-SUFFIX,mirrors.ustc.edu.cn,DIRECT
  - DOMAIN-SUFFIX,npmmirror.com,DIRECT
  - DOMAIN-SUFFIX,hf-mirror.com,DIRECT
```

如果是 WSL2 NAT 模式，可以补：

```yaml
- IP-CIDR,172.16.0.0/12,DIRECT
```

如果是 mirrored 模式，一般可以不用这条。

---

## 11. Clash 配置是否真正生效

### 11.1 查看 TUN 配置

需要 Clash API 9090 可用：

```bash
curl http://127.0.0.1:9090/configs 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
tun = d.get('tun', {})
print(json.dumps(tun, indent=2, ensure_ascii=False))
"
```

重点看：

```text
enable
stack
dns-hijack
inet4-address
inet6-address
auto-route
auto-detect-interface
```

如果只有：

```json
"inet4-address": ["198.18.0.1/30"]
```

说明 TUN 主要接管 IPv4。

---

### 11.2 查看 DNS 配置是否加载

```bash
curl http://127.0.0.1:9090/configs 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
dns = d.get('dns', 'NO DNS KEY')
print(json.dumps(dns, indent=2, ensure_ascii=False))
"
```

如果输出：

```text
NO DNS KEY
```

说明 DNS Mixin / 覆写没有加载成功。

---

### 11.3 查看 fake-ip-filter 是否生效

```bash
curl http://127.0.0.1:9090/configs 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
dns = d.get('dns', {})
print('enable:', dns.get('enable'))
print('enhanced-mode:', dns.get('enhanced-mode'))
print('listen:', dns.get('listen'))
print('fake-ip-filter count:', len(dns.get('fake-ip-filter', [])))
print('fake-ip-filter:', dns.get('fake-ip-filter'))
"
```

判断标准：

```text
DNS 配置不是 NO DNS KEY
fake-ip-filter 条数明显大于默认值
国内镜像在 filter 中
```

---

## 12. 网络诊断命令

### 12.1 WSL2 DNS 检查

```bash
cat /etc/resolv.conf

getent hosts mirrors.tuna.tsinghua.edu.cn
getent ahosts mirrors.tuna.tsinghua.edu.cn

dig mirrors.tuna.tsinghua.edu.cn
dig mirrors.tuna.tsinghua.edu.cn @127.0.0.1 -p 1053
```

判断：

```text
没有结果：
  DNS 失败

返回 198.18.x.x：
  走了 fake-ip

国内镜像返回真实 IP：
  fake-ip-filter 生效
```

---

### 12.2 IPv4 / IPv6 检查

```bash
curl -4 -I https://mirrors.tuna.tsinghua.edu.cn
curl -6 -I https://mirrors.tuna.tsinghua.edu.cn
```

判断：

```text
curl -4 正常，curl -6 失败：
  IPv6 不可用，应该禁用或降低 IPv6 优先级

curl -6 正常，curl -4 失败：
  IPv4/TUN 路径可能有问题

二者都失败：
  DNS、路由、代理或目标站点问题
```

---

### 12.3 代理端口检查

```bash
curl -I --proxy http://127.0.0.1:7890 https://github.com
curl -I --proxy http://127.0.0.1:7890 https://www.google.com
```

如果失败，检查：

```text
Clash 是否运行
端口是否是 7890 还是 7897
Clash 是否允许本地连接
WSL mirrored 下 127.0.0.1 是否可达 Windows Clash
```

---

### 12.4 apt 检查

```bash
sudo apt-get update
sudo apt-get -o Acquire::ForceIPv4=true update
proxysudo apt-get update
```

判断：

```text
ForceIPv4 成功：
  IPv6 问题

proxysudo 成功：
  TUN / DNS / 直连路径问题，7890 代理路径正常

全部失败：
  DNS 或 Clash 端口本身可能有问题
```

---

## 13. 网络缓存清理

Windows：

```powershell
ipconfig /flushdns
```

WSL：

```bash
sudo resolvectl flush-caches
```

如果 resolvectl 不存在，可以跳过。

apt：

```bash
sudo rm -rf /var/lib/apt/lists/*
sudo apt clean
sudo apt-get update
```

Clash：

```text
重启 Clash Verge Rev
重新加载配置
确认 Mixin / DNS 覆写生效
```

---

# 14. 最终推荐方案

## 14.1 Windows 侧

```text
继续使用 Clash TUN。
适合 ChatGPT、浏览器、QUIC、UDP 等复杂流量。
```

Clash 里：

```text
关闭 IPv6，前提是当前机器 IPv6 不可用
设置 fake-ip-filter
确保国内镜像域名返回真实 IP
确保 Clash API 9090 可用，便于诊断
```

---

## 14.2 WSL2 侧

推荐主策略：

```text
WSL2 不依赖 Windows TUN 接管。
WSL2 开发工具显式走 7890 代理端口。
```

也就是：

```bash
proxy_on
proxysudo apt-get update
proxy_run git clone ...
proxy_run curl ...
```

对于国内源：

```text
能直连就直连；
直连不稳时再走 7890；
不要同时让 apt 写死代理 + TUN + fake-ip 多层叠加。
```

---

## 14.3 DNS 侧

如果 WSL DNS 经常坏：

```bash
sudo nano /etc/wsl.conf
```

```ini
[network]
generateResolvConf = false
```

然后固定 IPv4 DNS：

```bash
sudo rm -f /etc/resolv.conf
sudo bash -c 'echo "nameserver 114.114.114.114" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 223.5.5.5" >> /etc/resolv.conf'
sudo bash -c 'echo "nameserver 119.29.29.29" >> /etc/resolv.conf'
```

必要时：

```bash
sudo chattr +i /etc/resolv.conf
```

恢复时：

```bash
sudo chattr -i /etc/resolv.conf
```

---

# 15. 一句话总结

你的问题根源可以归纳为：

```text
Windows TUN 对 Windows 本机是好东西；
但 WSL2 mirrored 下，Windows TUN 接管 WSL2 的 IPv4 流量时，
可能出现 DNS、fake-ip、TCP 连接上下文和 IPv4/IPv6 路由的叠加问题。

因此：
Windows 继续 TUN；
WSL2 开发工具走 7890；
国内镜像加入 fake-ip-filter 并优先真实 IP；
无 IPv6 机器关闭或降低 IPv6 优先级；
apt DNS 失败时固定 /etc/resolv.conf 为 IPv4 DNS。
```

最稳的工作模式是：

```text
Windows:
  TUN

WSL2:
  mirrored + 7890 显式代理 + IPv4 DNS + 必要时 ForceIPv4
```


Windows / WSL / Ubuntu 的 IPv4、IPv6 优先级与限制方法
0. 先说结论
在你这种场景里，问题通常不是“有没有 IPv6”，而是：
系统优先尝试 IPv6
↓
IPv6 路径不可用、不稳定、被 Clash TUN/Fake-IP/DNS 处理异常
↓
curl / pip / uv / apt 出现超时、TLS handshake eof、schannel failed
所以排查思路是：
1. 先测试 IPv4 / IPv6 哪个坏
2. 再临时强制某个命令使用 IPv4
3. 如果确认 IPv6 确实不稳定，再调整系统优先级
4. 最后才考虑彻底禁用 IPv6
一般不建议一上来全局禁用 IPv6。更推荐：
优先使用 IPv4
保留 IPv6
必要时只在 Clash / apt / curl / uv 场景中强制 IPv4

---
1. 通用诊断：判断是不是 IPv6 导致
Windows PowerShell
curl.exe -4 -I https://pypi.org/simple/pillow/
curl.exe -6 -I https://pypi.org/simple/pillow/

curl.exe -4 -I https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -6 -I https://mirrors.aliyun.com/pypi/simple/pillow/

curl.exe -4 -I https://pypi.tuna.tsinghua.edu.cn/simple/pillow/
curl.exe -6 -I https://pypi.tuna.tsinghua.edu.cn/simple/pillow/
如果结果是：
curl -4 正常
curl -6 失败
说明 IPv6 链路可能有问题。
如果结果是：
curl -4 正常
curl 默认失败
说明系统或程序默认优先走了有问题的 IPv6。

---
WSL / Ubuntu
curl -4 -I https://pypi.org/simple/pillow/
curl -6 -I https://pypi.org/simple/pillow/

curl -4 -I https://mirrors.aliyun.com/pypi/simple/pillow/
curl -6 -I https://mirrors.aliyun.com/pypi/simple/pillow/
也可以看解析结果：
getent ahosts pypi.org
如果优先出现 IPv6 地址，说明解析和连接可能优先走 IPv6。

---
2. Windows：查看 IPv4 / IPv6 是否启用
查看网卡协议绑定
管理员 PowerShell：
Get-NetAdapterBinding -Name "*" | Where-Object {
    $_.ComponentID -in "ms_tcpip", "ms_tcpip6"
} | Format-Table Name, DisplayName, ComponentID, Enabled
其中：
ms_tcpip   = IPv4
ms_tcpip6  = IPv6
正常情况下，当前正在使用的网卡至少要：
ms_tcpip   True
如果 IPv4 没开，必须启用。

---
启用 IPv4
先看网卡名：
Get-NetAdapter
假设网卡名是 以太网：
Enable-NetAdapterBinding -Name "以太网" -ComponentID ms_tcpip
如果是 Ethernet：
Enable-NetAdapterBinding -Name "Ethernet" -ComponentID ms_tcpip

---
临时禁用某张网卡的 IPv6
不建议一开始就做，但如果你已经确认 IPv6 出问题，可以对当前物理网卡禁用 IPv6。
Disable-NetAdapterBinding -Name "以太网" -ComponentID ms_tcpip6
恢复：
Enable-NetAdapterBinding -Name "以太网" -ComponentID ms_tcpip6
如果网卡名是英文：
Disable-NetAdapterBinding -Name "Ethernet" -ComponentID ms_tcpip6
Enable-NetAdapterBinding -Name "Ethernet" -ComponentID ms_tcpip6
注意：
这只是禁用某张网卡上的 IPv6，不是修改整个 Windows 的 IPv6 协议栈。

---
3. Windows：优先 IPv4，而不是彻底禁 IPv6
更推荐的方式是：让 Windows 优先使用 IPv4，但保留 IPv6。
Windows 使用前缀策略表控制 IPv4 / IPv6 优先级。
查看当前策略：
netsh interface ipv6 show prefixpolicies
你会看到类似：
Precedence  Label  Prefix
----------  -----  ------------------------
        50      0  ::1/128
        40      1  ::/0
        35      4  ::ffff:0:0/96
其中：
::/0             表示普通 IPv6
::ffff:0:0/96    表示 IPv4-mapped IPv6，也就是让 IPv4 更靠前的一类策略
提高 IPv4 优先级
管理员 PowerShell 执行：
netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4
然后再查看：
netsh interface ipv6 show prefixpolicies
如果 ::ffff:0:0/96 的优先级比 ::/0 高，那么系统会更倾向于 IPv4。

---
恢复默认前缀策略
如果后续想恢复：
netsh interface ipv6 reset
然后重启电脑。

---
4. Windows：不推荐但可用的全局禁用 IPv6
一般不推荐，除非你确认整个网络环境完全不需要 IPv6，且 IPv6 明确导致故障。
查看当前注册表项：
reg query "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip6\Parameters" /v DisabledComponents
设置为优先 IPv4，一般用：
reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip6\Parameters" /v DisabledComponents /t REG_DWORD /d 32 /f
设置后重启电脑。
如果要彻底禁用 IPv6，常见值是：
reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip6\Parameters" /v DisabledComponents /t REG_DWORD /d 255 /f
但这个不推荐。更安全的是只设置优先 IPv4：
DisabledComponents = 32
恢复默认：
reg delete "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip6\Parameters" /v DisabledComponents /f
然后重启电脑。

---
5. Clash Verge Dev 里的 IPv6 建议
如果你发现：
curl -4 正常
curl -6 失败
开启 Clash/TUN 后网络异常
DNS 出现 198.18.x.x fake-ip
建议 Clash 里先这样处理：
ipv6: false
DNS 部分也建议：
dns:
  enable: true
  ipv6: false
  enhanced-mode: redir-host
开发环境优先稳定的话，建议：
TUN: 关闭
System Proxy: 开启
DNS: redir-host
IPv6: false
如果你必须开 TUN，也建议先关闭 IPv6 和 Fake-IP：
ipv6: false

dns:
  enable: true
  ipv6: false
  enhanced-mode: redir-host

---
6. WSL2：先搞清楚网络模式
WSL2 有几种网络模式，常见是：
NAT 模式
Mirrored 模式
你之前提到过 WSL 是 mirrored，这种情况下 WSL 与 Windows 主机网络关系更紧密，Windows 侧的代理、DNS、IPv6、Clash TUN 更容易影响 WSL。
查看 WSL 配置文件：
Windows 用户目录下：
notepad $env:USERPROFILE\.wslconfig
可能有：
[wsl2]
networkingMode=mirrored
或者没有这个字段，就是默认 NAT 模式。
修改 .wslconfig 后，需要重启 WSL：
wsl --shutdown
然后重新打开 WSL。

---
7. WSL / Ubuntu：临时强制命令走 IPv4
curl
curl -4 -I https://pypi.org/simple/pillow/
apt
临时：
sudo apt -o Acquire::ForceIPv4=true update
安装：
sudo apt -o Acquire::ForceIPv4=true install <package>
例如：
sudo apt -o Acquire::ForceIPv4=true install build-essential

---
8. WSL / Ubuntu：永久让 apt 使用 IPv4
创建配置文件：
sudo nano /etc/apt/apt.conf.d/99force-ipv4
写入：
Acquire::ForceIPv4 "true";
保存后：
sudo apt update
如果要撤销：
sudo rm /etc/apt/apt.conf.d/99force-ipv4

---
9. Ubuntu/Linux：调整 IPv4 / IPv6 解析优先级
Linux 上通常通过 /etc/gai.conf 调整地址选择策略。
打开：
sudo nano /etc/gai.conf
找到这一行，取消注释：
precedence ::ffff:0:0/96  100
如果没有就手动添加：
precedence ::ffff:0:0/96  100
含义是：
优先使用 IPv4-mapped IPv6 地址
也就是更倾向 IPv4
保存后，新开的程序会更倾向 IPv4。为了保险，可以重启 WSL：
wsl --shutdown
然后重新进入 Ubuntu。

---
10. Ubuntu/Linux：临时禁用 IPv6
临时禁用当前 Linux 系统 IPv6：
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1
恢复：
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=0
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=0
查看状态：
cat /proc/sys/net/ipv6/conf/all/disable_ipv6
cat /proc/sys/net/ipv6/conf/default/disable_ipv6
含义：
0 = IPv6 启用
1 = IPv6 禁用

---
11. Ubuntu/Linux：永久禁用 IPv6
不推荐作为第一选择。如果确认必须禁用，可以：
sudo nano /etc/sysctl.d/99-disable-ipv6.conf
写入：
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
应用：
sudo sysctl --system
恢复时删除这个文件：
sudo rm /etc/sysctl.d/99-disable-ipv6.conf
sudo sysctl --system
或者重启 WSL：
wsl --shutdown

---
12. WSL 中 DNS 可能被 Windows / Clash 改乱
查看 WSL DNS：
cat /etc/resolv.conf
如果你看到类似：
nameserver 172.x.x.x
通常是 WSL 自动生成的 DNS。
如果看到：
nameserver 198.18.x.x
可能和 Clash Fake-IP / TUN 有关。
如果 WSL 里 apt update、curl 经常异常，可以临时测试：
nslookup pypi.org
nslookup github.com
curl -4 -I https://pypi.org/simple/pillow/
curl -6 -I https://pypi.org/simple/pillow/

---
13. WSL：手动固定 DNS
如果 DNS 明显异常，可以让 WSL 不自动生成 /etc/resolv.conf。
编辑：
sudo nano /etc/wsl.conf
写入：
[network]
generateResolvConf = false
删除旧的 resolv.conf：
sudo rm /etc/resolv.conf
重新创建：
sudo nano /etc/resolv.conf
写入：
nameserver 223.5.5.5
nameserver 119.29.29.29
nameserver 8.8.8.8
然后在 Windows PowerShell 执行：
wsl --shutdown
重新进入 WSL 测试：
nslookup pypi.org
curl -4 -I https://pypi.org/simple/pillow/
sudo apt update
如果你使用 Clash TUN/Fake-IP，这种固定 DNS 有时会绕开 Clash DNS，有利有弊。开发环境里通常更稳定，但某些代理规则可能不再按预期解析。

---
14. WSL 使用 Windows Clash 代理
如果不开 TUN，只开 Windows Clash 的系统代理，WSL 默认不一定自动走代理。
NAT 模式下
WSL 里获取 Windows 主机 IP：
cat /etc/resolv.conf | grep nameserver
假设结果是：
nameserver 172.25.64.1
那么可以设置：
export http_proxy=http://172.25.64.1:7890
export https_proxy=http://172.25.64.1:7890
测试：
curl -I https://pypi.org/simple/pillow/
Mirrored 模式下
有时可以直接用：
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
测试：
curl -I https://pypi.org/simple/pillow/
如果 127.0.0.1 不行，再用 Windows 主机 IP。

---
15. pip / uv / Python 层面的 IPv4 处理
curl 和 apt 可以直接 -4 或 ForceIPv4，但 pip / uv 没有特别常用的 -4 参数。
所以对于 uv / pip，更建议从系统层处理：
Windows：提高 IPv4 优先级
WSL/Ubuntu：修改 /etc/gai.conf
Clash：关闭 IPv6
DNS：避免 Fake-IP 冲突
如果只是临时安装，最稳是：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:UV_INDEX_URL="https://pypi.org/simple"

uv pip install dashscope pillow --link-mode=copy
或者走国内源直连：
$env:UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
uv pip install dashscope pillow --link-mode=copy

---
16. 推荐配置组合
方案 A：最稳开发配置，推荐
适合你现在这种 uv / pip / curl / apt 经常受 Clash 影响的情况。
Windows：
Clash 开启
System Proxy 开启
TUN 关闭
Clash IPv6 关闭
DNS 使用 redir-host
Windows 优先 IPv4
PowerShell：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
WSL/Ubuntu：
# /etc/gai.conf
precedence ::ffff:0:0/96  100
apt：
sudo nano /etc/apt/apt.conf.d/99force-ipv4
写入：
Acquire::ForceIPv4 "true";

---
方案 B：保留 IPv6，但优先 IPv4
Windows：
netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4
Ubuntu/WSL：
sudo nano /etc/gai.conf
添加：
precedence ::ffff:0:0/96  100
Clash：
ipv6: false
dns:
  ipv6: false
  enhanced-mode: redir-host
这套方案比较温和，推荐优先使用。

---
方案 C：彻底限制机器使用 IPv4
只有在你确认 IPv6 确实导致大量问题时再用。
Windows 禁用当前网卡 IPv6：
Disable-NetAdapterBinding -Name "以太网" -ComponentID ms_tcpip6
恢复：
Enable-NetAdapterBinding -Name "以太网" -ComponentID ms_tcpip6
Ubuntu/WSL 永久禁用 IPv6：
sudo nano /etc/sysctl.d/99-disable-ipv6.conf
写入：
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
应用：
sudo sysctl --system

---
17. 推荐你现在实际执行的顺序
第一步：确认是不是 IPv6 问题
Windows：
curl.exe -4 -I https://pypi.org/simple/pillow/
curl.exe -6 -I https://pypi.org/simple/pillow/
curl.exe -4 -I https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -6 -I https://mirrors.aliyun.com/pypi/simple/pillow/
WSL：
curl -4 -I https://pypi.org/simple/pillow/
curl -6 -I https://pypi.org/simple/pillow/

---
第二步：Windows 优先 IPv4
管理员 PowerShell：
netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4
查看：
netsh interface ipv6 show prefixpolicies

---
第三步：WSL/Ubuntu 优先 IPv4
sudo nano /etc/gai.conf
添加或取消注释：
precedence ::ffff:0:0/96  100
然后 Windows 侧：
wsl --shutdown
重新打开 WSL。

---
第四步：apt 强制 IPv4
sudo nano /etc/apt/apt.conf.d/99force-ipv4
写入：
Acquire::ForceIPv4 "true";
测试：
sudo apt update

---
第五步：Clash 关闭 IPv6
在 Clash 配置里优先使用：
ipv6: false

dns:
  enable: true
  ipv6: false
  enhanced-mode: redir-host
同时建议：
先关闭 TUN
只使用 System Proxy

---
18. 最后总结
你现在要解决的是：
Windows / WSL / Ubuntu 三层网络环境里，IPv6 路径可能不稳定；
Clash TUN / Fake-IP / DNS 又可能放大这个问题；
于是 uv、pip、curl、apt 出现间歇性失败。
推荐策略不是一刀切禁用 IPv6，而是：
Windows：优先 IPv4
WSL/Ubuntu：/etc/gai.conf 优先 IPv4
apt：必要时 ForceIPv4
Clash：关闭 IPv6，DNS 用 redir-host
TUN：先关闭，稳定后再尝试开启
最稳组合是：
Windows 优先 IPv4
Clash 不开 TUN
Clash IPv6=false
Clash DNS=redir-host
PowerShell 显式 HTTP_PROXY / HTTPS_PROXY
WSL /etc/gai.conf 优先 IPv4
apt 强制 IPv4


0. 问题背景
在 Windows 机器上使用 Clash Verge Dev 时，可能出现以下现象：
uv / pip / curl / 浏览器访问异常
访问 PyPI、清华源、知乎、百度等网站表现不一致
某些网站能打开，某些网站打不开
ping 通，但网页打不开
curl 报 TLS handshake / schannel 错误
关闭 Clash Verge Dev 后网络恢复正常
典型报错包括：
curl: (35) schannel: failed to receive handshake, SSL/TLS connection failed
或者：
tls handshake eof
这种问题通常不是 Python、conda、uv、pip、pillow、dashscope 本身的问题，而是 Clash 改变了 Windows 的网络路径后，导致 DNS、代理、TUN、IPv6 或规则分流出现冲突。

---
1. 先明确一个核心原则
不要一开始就同时修改很多配置。
正确排查思路是：
先建立基准线
再逐层打开 Clash 功能
观察是哪一层开始出问题
也就是按照：
完全关闭 Clash
↓
只开启 Clash 本地代理端口
↓
开启系统代理
↓
开启 TUN
↓
检查 DNS / Fake-IP / IPv6 / 规则
这样逐步排查。

---
2. ping 不是 HTTPS 排查工具
错误写法：
ping https://pypi.tuna.tsinghua.edu.cn
这是不对的，因为 ping 只能测试主机名或 IP，不能带 https://。
正确写法：
ping pypi.tuna.tsinghua.edu.cn
ping mirrors.tuna.tsinghua.edu.cn
ping www.baidu.com
但要注意：
ping 通 ≠ HTTPS 网站能正常访问
ping 只能说明 DNS 和 ICMP 大致可用，而网页访问还需要：
DNS 解析
TCP 连接
TLS 握手
HTTP/HTTPS 请求
浏览器代理
证书校验
Clash 规则分流
Clash DNS / Fake-IP
所以排查 PyPI、GitHub、百度、知乎这类 HTTPS 网站时，curl 比 ping 更有意义。

---
3. 用 curl 判断 HTTPS 是否真的可用
测试 PyPI 官方源：
curl.exe -I https://pypi.org/simple/pillow/
测试阿里源：
curl.exe -I https://mirrors.aliyun.com/pypi/simple/pillow/
测试清华源：
curl.exe -I https://pypi.tuna.tsinghua.edu.cn/simple/pillow/
curl.exe -I https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
测试不经过代理：
curl.exe -I --noproxy "*" https://pypi.org/simple/pillow/
curl.exe -I --noproxy "*" https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -I --noproxy "*" https://pypi.tuna.tsinghua.edu.cn/simple/pillow/
判断逻辑：
curl 正常，浏览器不正常
=> 浏览器代理、浏览器 DNS、插件、缓存问题

curl 不正常，但 curl --noproxy 正常
=> Clash 系统代理或规则分流有问题

curl 和 curl --noproxy 都不正常
=> Windows 网络、DNS、TUN、运营商链路、防火墙等问题

只有某一个网站异常
=> 可能是目标站点策略、规则分流、IPv6、DNS 缓存或代理节点链路问题

---
4. 301 Moved Permanently 不是失败
如果看到：
HTTP/1.1 301 Moved Permanently
Location: https://pypi.tuna.tsinghua.edu.cn/simple/
这不是网络失败，而是正常重定向。
常见原因是 URL 末尾缺少 /。
例如：
curl.exe -I https://pypi.tuna.tsinghua.edu.cn/simple
可能返回 301。
更规范写法是：
curl.exe -I https://pypi.tuna.tsinghua.edu.cn/simple/
所以，pip / uv 配置镜像源时也建议带上末尾 /：
python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
查看 pip 当前配置：
python -m pip config list -v

---
5. 检查 PowerShell 是否被代理环境变量影响
先看当前 PowerShell 会话中是否有代理变量：
Get-ChildItem Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue
如果没有输出，说明当前会话没有这些代理变量。
如果看到类似：
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
ALL_PROXY=socks5://127.0.0.1:7890
说明 pip、uv、curl、Python 程序可能会主动走这个代理。
临时清除当前 PowerShell 会话代理：
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
然后重新测试：
curl.exe -I https://pypi.org/simple/pillow/
curl.exe -I https://mirrors.aliyun.com/pypi/simple/pillow/
如果需要强制当前 PowerShell 走 Clash 代理，可以显式设置：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
端口要改成 Clash Verge Dev 里的实际 mixed port。

---
6. 检查 WinHTTP 代理
执行：
netsh winhttp show proxy
如果输出：
直接访问(没有代理服务器)
说明 WinHTTP 层没有代理。
注意：
WinHTTP 代理 ≠ 浏览器系统代理
Chrome / Edge 通常看的是 Windows 用户层 Internet Settings，也就是 WinINET / 系统代理。
所以即使 WinHTTP 显示没有代理，浏览器仍然可能在走 Clash 系统代理。

---
7. 检查 Windows 浏览器系统代理
PowerShell 执行：
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' |
Select-Object ProxyEnable, ProxyServer, AutoConfigURL
也可以用注册表命令：
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyEnable
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer
如果看到：
ProxyEnable : 1
ProxyServer : 127.0.0.1:7890
说明 Windows 系统代理开启，浏览器会把流量交给本地 Clash 代理端口。
需要重点检查：
Clash Verge Dev 实际 mixed port 是多少
Windows 系统代理指向的端口是多少
二者是否一致
如果 Clash 实际端口是 7897，但系统代理还指向 7890，就可能出现：
部分网站打不开
部分网站卡住
书签栏网站能开
新访问的网站打不开
curl / 浏览器表现不一致

---
8. 确认 Clash 本地端口是否在监听
打开 Clash Verge Dev，找到端口配置：
HTTP Port
SOCKS Port
Mixed Port
System Proxy
TUN Mode
DNS
常见端口：
7890
7897
7899
假设 mixed port 是 7890，测试：
netstat -ano | findstr 7890
如果有输出，说明 Clash 本地代理端口正在监听。
如果没有输出，但系统代理却指向 127.0.0.1:7890，那网络大概率会异常。

---
9. 分模式排查 Clash
9.1 模式 A：完全关闭 Clash
这是基准线。
curl.exe -I https://pypi.org/simple/pillow/
curl.exe -I https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -I https://pypi.tuna.tsinghua.edu.cn/simple/pillow/
记录哪些能通。
如果关闭 Clash 后全部正常，说明问题基本在 Clash、TUN、DNS、规则或代理端口。

---
9.2 模式 B：只开 Clash，不开系统代理，不开 TUN
这个状态下 Clash 只是一个本地代理服务，Windows 流量不会自动走它。
手动让当前 PowerShell 走代理：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
测试：
curl.exe -I https://pypi.org/simple/pillow/
curl.exe -I https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -I https://pypi.tuna.tsinghua.edu.cn/simple/pillow/
如果这样正常，说明：
Clash 本体可用
代理端口可用
节点基本可用
此时可以尝试安装：
$env:UV_INDEX_URL="https://pypi.org/simple"
uv pip install dashscope pillow --link-mode=copy

---
9.3 模式 C：开启系统代理，但不开 TUN
这时浏览器和一部分命令行工具会走系统代理。
测试：
curl.exe -I https://pypi.org/simple/pillow/
uv pip install dashscope pillow --link-mode=copy -i https://pypi.org/simple
如果正常，说明这是比较稳定的开发模式。
如果浏览器正常，但 PowerShell 里的 curl、uv 不稳定，可以继续在 PowerShell 显式设置：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"

---
9.4 模式 D：开启 TUN
TUN 最容易出问题。
TUN 会创建虚拟网卡，接管系统流量、DNS、路由。它比普通系统代理更强，但也更容易和 Windows 网络栈冲突。
开启 TUN 后检查：
Get-NetAdapter
ipconfig /all
route print
Get-DnsClientServerAddress
重点看有没有类似：
Clash
Meta
Mihomo
TUN
Wintun
之类的虚拟网卡。
如果一开 TUN 就坏，问题通常集中在：
TUN 虚拟网卡
Clash DNS
Fake-IP
Service Mode
Windows 路由
防火墙
IPv6

---
10. 重点排查 TUN / DNS / Fake-IP
10.1 看 TUN 虚拟网卡是否正常
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed
如果 Clash TUN 网卡状态异常，例如：
Disconnected
Disabled
反复闪断
就可能导致网络不稳定。

---
10.2 检查 IPv4 / IPv6 绑定
Get-NetAdapterBinding -Name "*" | Where-Object {
    $_.ComponentID -in "ms_tcpip", "ms_tcpip6"
} | Format-Table Name, DisplayName, ComponentID, Enabled
正常情况下，正在使用的物理网卡至少要：
ms_tcpip   True
也就是 IPv4 启用。
如果你的机器、网络、代理配置主要走 IPv4，而 IPv6 状态混乱，可以测试：
curl.exe -4 -I https://pypi.org/simple/pillow/
curl.exe -6 -I https://pypi.org/simple/pillow/
如果：
curl -4 正常
curl -6 失败
说明 IPv6 路径可能有问题。
这时可以考虑在 Clash 里关闭 IPv6：
ipv6: false
或者在 Clash Verge Dev 设置里关闭 IPv6 相关选项。

---
10.3 检查 DNS 是否被 TUN 改乱
执行：
nslookup pypi.org
nslookup pypi.tuna.tsinghua.edu.cn
nslookup github.com
再看 DNS 服务器：
Get-DnsClientServerAddress
如果开 Clash 后 DNS 出现类似：
198.18.x.x
这通常是 Clash Fake-IP 机制。
Fake-IP 本身不是错误，但如果 Windows、TUN、某些开发工具链不能正确处理，就会出现：
能解析，但连接失败
TLS handshake failed
curl schannel failed
pip / uv 下载失败

---
11. Clash DNS 建议配置
如果只是为了开发环境稳定，不追求特别复杂的分流，建议先用保守 DNS 配置。
可以优先尝试：
dns:
  enable: true
  listen: 0.0.0.0:1053
  ipv6: false
  enhanced-mode: redir-host
  nameserver:
    - 223.5.5.5
    - 119.29.29.29
    - 8.8.8.8
  fallback:
    - 1.1.1.1
    - 8.8.8.8
重点是：
enhanced-mode: redir-host
先不要用：
enhanced-mode: fake-ip
因为 Fake-IP 在 Windows + TUN + 开发工具链里更容易引发奇怪问题。
如果必须使用 Fake-IP，至少加一些过滤：
fake-ip-filter:
  - "*.lan"
  - "*.local"
  - "dns.msftncsi.com"
  - "www.msftncsi.com"
  - "time.windows.com"
  - "pypi.org"
  - "*.pypi.org"
  - "*.pythonhosted.org"
  - "files.pythonhosted.org"
  - "github.com"
  - "*.github.com"
  - "*.githubusercontent.com"
  - "*.githubassets.com"
  - "*.tuna.tsinghua.edu.cn"
  - "*.aliyun.com"

---
12. Clash 规则建议
开发相关域名建议单独指定规则，避免被错误分流。
示例：
rules:
  - DOMAIN-SUFFIX,pypi.org,Proxy
  - DOMAIN-SUFFIX,pythonhosted.org,Proxy
  - DOMAIN-SUFFIX,files.pythonhosted.org,Proxy
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
注意规则顺序：
具体规则要放在 MATCH 之前
国内 DIRECT 规则要放在最终兜底规则之前
否则国内网站可能被错误丢到代理节点，表现为：
百度 / 知乎打不开
国内镜像源慢或失败
TLS 握手异常
连接超时

---
13. 浏览器 Secure DNS 也可能冲突
Chrome / Edge 里有“安全 DNS / Secure DNS”。
它可能让浏览器绕过系统 DNS，直接使用 DoH 解析域名，从而和 Clash 的 DNS / Fake-IP / 规则分流冲突。
Chrome：
chrome://settings/security
Edge：
edge://settings/privacy
找到：
使用安全 DNS / Use secure DNS
排查时可以先关闭，然后彻底重启浏览器。
如果关闭后浏览器恢复正常，而 curl 一直正常，说明问题主要在浏览器 DNS 行为。

---
14. Windows 网络栈重置
轻量操作：
ipconfig /flushdns
更重的操作：
netsh winsock reset
netsh int ip reset
后两条通常需要重启电脑。
推荐顺序：
先 flushdns
再重启浏览器
再重启 Clash Verge Dev
最后才考虑 winsock reset / ip reset / 重启电脑
如果 Clash TUN 把网络栈搞乱过，可以管理员 PowerShell 执行：
netsh winsock reset
netsh int ip reset
ipconfig /flushdns
然后重启电脑。
重启后先不要开 Clash，先测试：
curl.exe -I https://pypi.org/simple/pillow/
正常后再逐层开启 Clash。

---
15. uv / pip 安装建议
如果 Clash 不稳定，先绕开有问题的源。
官方 PyPI：
uv pip install dashscope pillow --link-mode=copy -i https://pypi.org/simple
阿里源：
uv pip install dashscope pillow --link-mode=copy -i https://mirrors.aliyun.com/pypi/simple/
当前 PowerShell 临时设置 uv 源：
$env:UV_INDEX_URL="https://pypi.org/simple"
uv pip install dashscope pillow --link-mode=copy
或者：
$env:UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
uv pip install dashscope pillow --link-mode=copy
如果需要当前 PowerShell 显式走 Clash：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:UV_INDEX_URL="https://pypi.org/simple"

uv pip install dashscope pillow --link-mode=copy
如果官方源慢，再换阿里源：
$env:UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
uv pip install dashscope pillow --link-mode=copy

---
16. 推荐的日常稳定配置
对开发环境来说，最稳的方式通常是：
不开 TUN
只开 System Proxy
PowerShell 需要时显式设置 HTTP_PROXY / HTTPS_PROXY
DNS 使用 redir-host
IPv6 暂时关闭
PyPI / GitHub 规则单独指定
推荐状态：
Clash Verge Dev: 开启
System Proxy: 开启
TUN: 关闭
DNS enhanced-mode: redir-host
IPv6: 关闭
PowerShell: 显式设置代理变量
PowerShell：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:UV_INDEX_URL="https://pypi.org/simple"
然后安装：
uv pip install dashscope pillow --link-mode=copy

---
17. 一键诊断脚本
可以保存成 check_clash_network.ps1：
Write-Host "=== WinHTTP proxy ==="
netsh winhttp show proxy

Write-Host "`n=== PowerShell proxy env ==="
Get-ChildItem Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue

Write-Host "`n=== Windows Internet Settings proxy ==="
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' |
Select-Object ProxyEnable, ProxyServer, AutoConfigURL

Write-Host "`n=== Clash port check ==="
netstat -ano | findstr 7890

Write-Host "`n=== Network adapters ==="
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed

Write-Host "`n=== IPv4 / IPv6 bindings ==="
Get-NetAdapterBinding -Name "*" | Where-Object {
    $_.ComponentID -in "ms_tcpip", "ms_tcpip6"
} | Format-Table Name, DisplayName, ComponentID, Enabled

Write-Host "`n=== DNS servers ==="
Get-DnsClientServerAddress

Write-Host "`n=== DNS lookup ==="
nslookup pypi.org
nslookup pypi.tuna.tsinghua.edu.cn
nslookup mirrors.aliyun.com
nslookup github.com
nslookup www.baidu.com
nslookup www.zhihu.com

Write-Host "`n=== curl PyPI mirrors ==="
curl.exe -I https://pypi.org/simple/pillow/
curl.exe -I https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -I https://pypi.tuna.tsinghua.edu.cn/simple/pillow/

Write-Host "`n=== curl PyPI mirrors no proxy ==="
curl.exe -I --noproxy "*" https://pypi.org/simple/pillow/
curl.exe -I --noproxy "*" https://mirrors.aliyun.com/pypi/simple/pillow/
curl.exe -I --noproxy "*" https://pypi.tuna.tsinghua.edu.cn/simple/pillow/

Write-Host "`n=== curl common sites ==="
curl.exe -I https://www.baidu.com
curl.exe -I https://www.zhihu.com

Write-Host "`n=== curl common sites no proxy ==="
curl.exe -I --noproxy "*" https://www.baidu.com
curl.exe -I --noproxy "*" https://www.zhihu.com

Write-Host "`n=== curl IPv4 / IPv6 ==="
curl.exe -4 -I https://pypi.org/simple/pillow/
curl.exe -6 -I https://pypi.org/simple/pillow/

Write-Host "`n=== pip config ==="
python -m pip config list -v
如果你的 Clash mixed port 不是 7890，记得把脚本里的：
netstat -ano | findstr 7890
改成对应端口，例如：
netstat -ano | findstr 7897

---
18. 推荐排查顺序
以后再遇到类似问题，按这个顺序：
1. 不要先改一堆配置，先用 curl 判断 HTTPS 是否通。
2. 用 curl --noproxy "*" 判断是否是代理导致。
3. 检查 PowerShell 环境变量代理。
4. 检查 WinHTTP 代理。
5. 检查 Windows 浏览器系统代理。
6. 检查 Clash 端口是否和系统代理一致。
7. 关闭浏览器 Secure DNS。
8. 临时关闭 Clash TUN，只保留系统代理测试。
9. 检查 Clash 规则，国内站点是否 DIRECT。
10. 测试 IPv4 / IPv6。
11. 检查 DNS 是否出现 198.18.x.x Fake-IP。
12. flushdns，重启浏览器，必要时重启 Clash。
13. 最后再考虑 winsock reset / ip reset / 重启电脑。

---
19. 本次问题的结论
根据你的现象：
关闭 Clash Verge Dev 后，uv / curl / PyPI 恢复正常
所以结论是：
这不是 conda / uv / pillow / dashscope 的问题。
也不是单纯的 Python 包源问题。
更像是 Clash Verge Dev 开启后，Windows 的 DNS、系统代理、TUN 虚拟网卡、IPv6 或规则分流改变了 HTTPS 连接路径，导致 PyPI 镜像站 TLS 握手失败。
优先怀疑：
TUN 模式
Fake-IP DNS
IPv6 路径
系统代理端口不一致
PyPI / 国内镜像源被错误分流
浏览器 Secure DNS
Windows 网络栈缓存状态异常

---
20. 最小稳定方案
如果你只是为了稳定开发，不想折腾复杂透明代理，建议先采用这个方案：
Clash Verge Dev 开启
TUN 关闭
System Proxy 开启
DNS 使用 redir-host
IPv6 关闭
PowerShell 显式设置 HTTP_PROXY / HTTPS_PROXY
PyPI 官方源走 Proxy，国内镜像源走 DIRECT
PowerShell 示例：
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:UV_INDEX_URL="https://pypi.org/simple"

uv pip install dashscope pillow --link-mode=copy
如果官方 PyPI 慢或不稳：
$env:UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
uv pip install dashscope pillow --link-mode=copy
一句话总结：
开发环境优先求稳定：先不要开 TUN。
系统代理 + PowerShell 显式代理变量 + redir-host DNS，通常比 TUN + Fake-IP 更稳。




