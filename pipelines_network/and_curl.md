# curl 指令指南：Windows / Ubuntu / WSL 网络测试手册

## 这份笔记解决什么

`curl` 的核心能力是：从命令行发起网络请求，并把请求过程、响应头、响应体、TLS、代理、DNS、耗时等信息暴露出来。

它既可以当下载工具、API 调试工具，也很适合排查这种问题：

```text
WSL curl -> DNS / IPv4 / IPv6 / 代理 / TUN / 防火墙 / Clash -> 目标 HTTPS 服务
```

参考材料：

- 本地资料：`D:\ddesktop\FileParser\sources\Linux curl 命令 _ 菜鸟教程.html`
- 结合案例：WSL + Clash Verge Rev TUN 下更新 `opencode` 遇到 TLS EOF

## 0. Windows 和 Ubuntu/WSL 的差异

### 命令名称

Windows PowerShell 里建议明确使用：

```powershell
curl.exe https://example.com
```

原因：旧版 PowerShell 里 `curl` 可能是 `Invoke-WebRequest` 的别名。用 `curl.exe` 可以确保调用真正的 curl。

Ubuntu / WSL：

```bash
curl https://example.com
```

### 换行写法

PowerShell 使用反引号续行：

```powershell
curl.exe -X POST `
  -H "Content-Type: application/json" `
  -d '{"name":"alice"}' `
  https://httpbin.org/post
```

Bash / Ubuntu / WSL 使用反斜杠续行：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"alice"}' \
  https://httpbin.org/post
```

### 输出到空设备

Windows：

```powershell
curl.exe -o NUL -s -w "total=%{time_total}`n" https://example.com
```

Ubuntu / WSL：

```bash
curl -o /dev/null -s -w "total=%{time_total}\n" https://example.com
```

### 证书栈差异

Windows 版 curl 常见 TLS 错误可能出现 `schannel` 字样：

```text
curl: (35) schannel: failed to receive handshake
```

Ubuntu / WSL 通常基于 OpenSSL，错误可能像：

```text
curl: (35) error:0A000126:SSL routines::unexpected eof while reading
```

这两个都可能说明 TLS 握手或传输被中间链路提前断开。

## 1. 基础请求

### GET 请求

```bash
curl https://example.com
```

只看响应头：

```bash
curl -I https://example.com
```

响应头和响应体都显示：

```bash
curl -i https://example.com
```

跟随重定向：

```bash
curl -L https://example.com
```

常用组合：

```bash
curl -iL https://example.com
```

## 2. 输出控制

保存为指定文件：

```bash
curl -o output.html https://example.com
```

按远程文件名保存：

```bash
curl -O https://example.com/file.zip
```

静默模式：

```bash
curl -s https://example.com
```

静默但保留错误：

```bash
curl -sS https://example.com
```

失败时返回非 0 退出码，适合脚本：

```bash
curl -f https://example.com
```

脚本下载常用组合：

```bash
curl -fsSL https://example.com/install.sh | bash
```

含义：

| 参数 | 作用 |
| --- | --- |
| `-f` | HTTP 4xx/5xx 时失败退出 |
| `-s` | 静默模式 |
| `-S` | 静默时仍显示错误 |
| `-L` | 跟随重定向 |

## 3. 详细调试

显示完整请求过程：

```bash
curl -v https://example.com
```

更详细，包括传输字节：

```bash
curl --trace-ascii trace.txt https://example.com
```

只看连接和响应耗时：

Ubuntu / WSL：

```bash
curl -o /dev/null -s -w "dns=%{time_namelookup} connect=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total}\n" https://example.com
```

Windows：

```powershell
curl.exe -o NUL -s -w "dns=%{time_namelookup} connect=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total}`n" https://example.com
```

常用耗时字段：

| 字段 | 含义 |
| --- | --- |
| `time_namelookup` | DNS 解析耗时 |
| `time_connect` | TCP 连接完成耗时 |
| `time_appconnect` | TLS 握手完成耗时 |
| `time_starttransfer` | 首字节返回耗时 |
| `time_total` | 总耗时 |

## 4. HTTP 方法和请求体

POST 表单：

```bash
curl -X POST -d "username=admin&password=123456" https://httpbin.org/post
```

POST JSON：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"alice","age":20}' \
  https://httpbin.org/post
```

从文件读取 JSON：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  --data @payload.json \
  https://httpbin.org/post
```

PUT：

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"name":"new"}' https://httpbin.org/put
```

DELETE：

```bash
curl -X DELETE https://httpbin.org/delete
```

## 5. 请求头、认证、Cookie

添加请求头：

```bash
curl -H "Accept: application/json" https://api.github.com
```

Bearer Token：

```bash
curl -H "Authorization: Bearer <TOKEN>" https://api.example.com/me
```

Basic Auth：

```bash
curl -u username:password https://example.com/secure
```

发送 Cookie：

```bash
curl -b "session=abc123" https://example.com
```

保存 Cookie：

```bash
curl -c cookies.txt https://example.com/login
```

读取 Cookie：

```bash
curl -b cookies.txt https://example.com/profile
```

保存并读取 Cookie：

```bash
curl -c cookies.txt -b cookies.txt https://example.com
```

## 6. 文件上传下载

下载文件：

```bash
curl -O https://example.com/file.zip
```

断点续传：

```bash
curl -C - -O https://example.com/file.zip
```

限速下载：

```bash
curl --limit-rate 200K -O https://example.com/file.zip
```

上传表单文件：

```bash
curl -F "file=@localfile.txt" https://httpbin.org/post
```

多个表单字段：

```bash
curl -F "file=@localfile.txt" -F "name=test" https://httpbin.org/post
```

## 7. 代理相关

显式指定 HTTP 代理：

```bash
curl -x http://127.0.0.1:7897 https://api.github.com
```

显式指定 SOCKS5 代理：

```bash
curl -x socks5h://127.0.0.1:7897 https://api.github.com
```

说明：

| 写法 | DNS 由谁解析 |
| --- | --- |
| `socks5://host:port` | 本机 curl 解析 |
| `socks5h://host:port` | 代理端解析 |

临时设置代理环境变量：

Ubuntu / WSL：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897
```

Windows PowerShell：

```powershell
$env:http_proxy="http://127.0.0.1:7897"
$env:https_proxy="http://127.0.0.1:7897"
$env:all_proxy="http://127.0.0.1:7897"
```

清空代理环境变量：

Ubuntu / WSL：

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

Windows PowerShell：

```powershell
Remove-Item Env:http_proxy -ErrorAction SilentlyContinue
Remove-Item Env:https_proxy -ErrorAction SilentlyContinue
Remove-Item Env:all_proxy -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

查看代理环境变量：

Ubuntu / WSL：

```bash
env | grep -i proxy
```

Windows PowerShell：

```powershell
Get-ChildItem Env:*proxy*
```

## 8. DNS、IPv4、IPv6 测试

强制 IPv4：

```bash
curl -4 -i https://api.github.com/rate_limit
```

强制 IPv6：

```bash
curl -6 -i https://api.github.com/rate_limit
```

指定域名解析到某个 IP，不改系统 hosts：

```bash
curl --resolve example.com:443:93.184.216.34 https://example.com
```

连接某个 IP，但保留 Host/SNI：

```bash
curl --connect-to example.com:443:93.184.216.34:443 https://example.com
```

Ubuntu / WSL 查看解析：

```bash
getent ahosts api.github.com
```

Windows 查看解析：

```powershell
Resolve-DnsName api.github.com
```

## 9. TLS / 证书测试

忽略证书验证：

```bash
curl -k https://self-signed.example.com
```

注意：`-k` 只适合临时诊断，不要当成长期方案。

指定 CA 证书：

```bash
curl --cacert cert.pem https://secure.example.com
```

查看 TLS 握手过程：

```bash
curl -v https://example.com
```

常见错误判断：

| 错误 | 常见含义 |
| --- | --- |
| `Could not resolve host` | DNS 解析失败 |
| `Connection refused` | 目标端口无服务或被拒绝 |
| `Connection timed out` | 网络不通、被丢包、防火墙或路由问题 |
| `SSL certificate problem` | 证书链或 CA 问题 |
| `unexpected eof while reading` | TLS 连接被提前断开，常见于代理/TUN/出口链路 |
| `schannel: failed to receive handshake` | Windows TLS 握手失败，可能是代理或出口链路中断 |
| `HTTP/1.1 403` | 服务端返回拒绝，已经到达 HTTP 层 |
| `HTTP/1.1 429` | 请求过多，服务端限流 |

## 10. 网络排障最小模板

### Ubuntu / WSL

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "===== DNS ====="
getent ahosts api.github.com

echo "===== IPv4 ====="
curl -4 -i --max-time 20 https://api.github.com/rate_limit

echo "===== IPv6 ====="
curl -6 -i --max-time 20 https://api.github.com/rate_limit

echo "===== Proxy Env ====="
env | grep -i proxy
```

### Windows PowerShell

```powershell
Remove-Item Env:http_proxy -ErrorAction SilentlyContinue
Remove-Item Env:https_proxy -ErrorAction SilentlyContinue
Remove-Item Env:all_proxy -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue

Write-Host "===== DNS ====="
Resolve-DnsName api.github.com

Write-Host "===== IPv4 ====="
curl.exe -4 -i --max-time 20 https://api.github.com/rate_limit

Write-Host "===== IPv6 ====="
curl.exe -6 -i --max-time 20 https://api.github.com/rate_limit

Write-Host "===== Proxy Env ====="
Get-ChildItem Env:*proxy*
```

判断方式：

| 现象 | 说明 |
| --- | --- |
| `curl -4` 成功，`curl -6` 失败 | IPv6 链路问题，后续命令可先加 `-4` |
| `curl -4` 也 TLS EOF | 代理、TUN、节点、TLS 链路可能提前中断 |
| 清空代理后成功 | 原代理环境变量配置有问题 |
| 显式 `-x` 代理成功 | 透明代理/TUN 路径有问题，但 HTTP 代理端口可用 |
| 关闭代理软件后成功 | 代理软件的 TUN、DNS、规则或节点配置冲突 |
| 返回 `403` / `429` | 已到达 HTTP 层，问题转为权限、限流或服务端策略 |

## 11. Clash Verge Rev / WSL 专用模板

### 1. 看 WSL DNS 是否被 fake-ip 接管

```bash
getent ahosts api.github.com
```

如果看到类似：

```text
198.18.0.41
```

通常表示 Clash fake-ip/TUN 正在接管解析。

### 2. 测试不显式代理的路径

```bash
curl -4 -i --max-time 20 https://api.github.com/rate_limit
```

如果出现：

```text
curl: (35) error:0A000126:SSL routines::unexpected eof while reading
```

说明 HTTPS 在 TLS 握手或传输阶段被提前断开。

### 3. 测试显式走 Clash HTTP 代理

常见端口：

```bash
curl -x http://127.0.0.1:7897 -4 -i --max-time 20 https://api.github.com/rate_limit
```

如果返回：

```text
HTTP/1.1 200 Connection established
HTTP/2 200
```

说明 Clash HTTP 代理端口是通的，问题集中在 TUN/fake-ip 透明代理路径。

### 4. 给当前 WSL shell 设置代理

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897
```

然后执行安装或升级：

```bash
curl -4 -fsSL https://opencode.ai/install | bash
```

或：

```bash
opencode upgrade
```

### 5. 如果 127.0.0.1 不通

查看 WSL 里的 Windows 主机 IP：

```bash
cat /etc/resolv.conf
```

尝试用 `nameserver` 地址：

```bash
curl -x http://<Windows主机IP>:7897 -4 -i https://api.github.com/rate_limit
```

成功后设置：

```bash
export http_proxy=http://<Windows主机IP>:7897
export https_proxy=http://<Windows主机IP>:7897
export all_proxy=http://<Windows主机IP>:7897
```

## 12. GitHub API 测试

查看匿名 API 限流：

```bash
curl -i https://api.github.com/rate_limit
```

重点看：

```text
X-RateLimit-Limit
X-RateLimit-Remaining
X-RateLimit-Reset
```

如果：

```text
X-RateLimit-Remaining: 0
```

说明匿名额度已耗尽。网络通了以后，依旧可能因为 GitHub API 限流导致安装器或升级器失败。

带 GitHub token 测试：

```bash
curl -H "Authorization: Bearer <GITHUB_TOKEN>" -i https://api.github.com/rate_limit
```

## 13. 常用组合速查

| 场景 | 命令 |
| --- | --- |
| 看网页响应头 | `curl -I https://example.com` |
| 看响应头和响应体 | `curl -i https://example.com` |
| 跟随跳转 | `curl -L https://example.com` |
| 调试 TLS/代理细节 | `curl -v https://example.com` |
| 下载脚本并执行 | `curl -fsSL https://example.com/install.sh \| bash` |
| 保存文件 | `curl -o file.txt https://example.com/file.txt` |
| 使用远程文件名保存 | `curl -O https://example.com/file.zip` |
| 强制 IPv4 | `curl -4 https://example.com` |
| 强制 IPv6 | `curl -6 https://example.com` |
| 使用 HTTP 代理 | `curl -x http://127.0.0.1:7897 https://example.com` |
| 忽略证书验证 | `curl -k https://example.com` |
| POST JSON | `curl -X POST -H "Content-Type: application/json" -d '{"a":1}' https://httpbin.org/post` |
| 上传文件 | `curl -F "file=@a.txt" https://httpbin.org/post` |
| 测耗时 | `curl -o /dev/null -s -w "total=%{time_total}\n" https://example.com` |

## 14. 我的排障顺序

遇到 curl 网络问题时，优先按这个顺序：

1. `curl -I` 看是否能到 HTTP 层。
2. `curl -v` 看卡在 DNS、TCP、TLS 还是 HTTP。
3. `curl -4` / `curl -6` 区分 IPv4 和 IPv6。
4. 查看代理环境变量。
5. 清空代理变量后重试。
6. 显式 `-x` 指定代理端口重试。
7. 检查 DNS 解析结果是否是 fake-ip。
8. 对比 Windows 侧和 WSL 侧结果。
9. 如果 HTTP 已返回 `403`、`429`，就从网络问题切换到权限、限流、服务端策略问题。

这次 `opencode` 的情况就是典型例子：不显式代理时 WSL 走 Clash TUN/fake-ip，TLS 提前 EOF；显式 `-x http://127.0.0.1:7897` 后可以到达 GitHub API，但又暴露了匿名 API 额度为 0 的问题。
