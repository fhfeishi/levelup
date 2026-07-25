# WSL + Clash Verge Rev TUN 下 opencode 更新 TLS EOF 排障笔记

## 背景

在 WSL 侧更新 `opencode` 时遇到 HTTPS/TLS 传输错误：

```bash
curl: (35) error:0A000126:SSL routines::unexpected eof while reading
```

Windows 侧之前也出现过类似错误：

```text
curl: (35) schannel: failed to receive handshake, SSL/TLS connection failed
```

这类错误通常不是 GitHub API 返回的 `403`，而是 HTTPS 连接在 TLS 握手或传输阶段被中间链路提前断开。

可能链路：

```text
WSL curl -> DNS / IPv4 / IPv6 / 代理环境变量 / Clash TUN / 防火墙 / 代理出口 -> GitHub TLS
```

## 已验证结果

在 WSL 中执行：

```bash
getent ahosts api.github.com
```

结果中 `api.github.com` 被解析到了：

```text
198.18.0.41
```

`198.18.0.0/15` 是 Clash fake-ip/TUN 常见地址段。这说明 WSL 当前 DNS/网络路径被 Clash TUN/fake-ip 接管。

继续测试：

```bash
curl -4 -i https://api.github.com/rate_limit
```

结果仍然失败：

```bash
curl: (35) error:0A000126:SSL routines::unexpected eof while reading
```

IPv6 测试：

```bash
curl -6 -i https://api.github.com/rate_limit
```

结果为无法连接：

```bash
curl: (7) Couldn't connect to server
```

代理环境变量为空：

```bash
env | grep -i proxy
```

没有输出，说明当时不是因为 shell 里的 `http_proxy` / `https_proxy` / `all_proxy` 环境变量导致。

## 关键结论

WSL 不显式指定代理时，流量走 Clash TUN/fake-ip 透明代理路径，这条路径上的 HTTPS 连接会被提前断开。

但显式走 Clash HTTP 代理端口是成功的：

```bash
curl -x http://127.0.0.1:7897 -4 -i https://api.github.com/rate_limit
```

返回成功：

```text
HTTP/1.1 200 Connection established
HTTP/2 200
```

所以问题不是 `opencode` 自身，也不是 `opencode.ai` 完全不可达，而是：

```text
WSL -> Clash TUN/fake-ip -> GitHub HTTPS
```

这条透明代理链路不稳定或配置冲突。

## 临时修复方案

在当前 WSL shell 中显式让命令走 Clash HTTP 代理：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897
```

然后重新安装或升级 `opencode`：

```bash
curl -4 -fsSL https://opencode.ai/install | bash
```

或者：

```bash
opencode upgrade
```

如果 `127.0.0.1:7897` 不通，可以查看 WSL 里的 Windows 主机 IP：

```bash
cat /etc/resolv.conf
```

找到 `nameserver` 对应地址后尝试：

```bash
curl -x http://<Windows主机IP>:7897 -4 -i https://api.github.com/rate_limit
```

如果成功，则改用：

```bash
export http_proxy=http://<Windows主机IP>:7897
export https_proxy=http://<Windows主机IP>:7897
export all_proxy=http://<Windows主机IP>:7897
```

## 最小诊断命令

以后遇到类似问题，可以先跑这一组：

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "===== DNS ====="
getent ahosts api.github.com

echo "===== IPv4 ====="
curl -4 -i https://api.github.com/rate_limit

echo "===== IPv6 ====="
curl -6 -i https://api.github.com/rate_limit

echo "===== Proxy Env ====="
env | grep -i proxy
```

判断方式：

| 现象 | 说明 |
| --- | --- |
| `curl -4` 成功，`curl -6` 失败 | IPv6 路径有问题，后续命令优先加 `-4` |
| `curl -4` 也 EOF | 代理、TUN、节点或 TLS 链路被中断 |
| 清空代理后成功 | 原代理环境变量配置有问题 |
| `curl -x http://127.0.0.1:7897` 成功 | WSL 需要显式走 Clash HTTP 代理 |
| 关闭 Clash 后成功 | Clash TUN、DNS、规则、fake-ip 或节点配置冲突 |

## GitHub API 限流情况

本次测试中 GitHub API 已经返回：

```text
X-RateLimit-Remaining: 0
```

这表示未认证 GitHub API 额度已经用完。网络链路修好后，如果安装器或升级器继续访问 GitHub API，仍可能遇到限流问题。

处理思路：

1. 等待 GitHub 匿名 API 额度恢复。
2. 给相关工具配置 GitHub token。
3. 尽量避免反复请求 GitHub API。

## 后续优化方向

可以优先检查 Clash Verge Rev：

1. TUN 模式是否和 WSL mirrored/networking 模式冲突。
2. fake-ip 是否导致 WSL 内解析到 `198.18.x.x` 后透明转发异常。
3. GitHub 相关规则是否走了不稳定节点。
4. IPv6 是否被优先使用但实际不可达。
5. DNS 覆写是否影响了 WSL 内的解析路径。

当前最实用的稳定方案是：在 WSL 中显式设置 `http_proxy`、`https_proxy`、`all_proxy` 到 `http://127.0.0.1:7897`，绕开 TUN/fake-ip 透明代理路径。
