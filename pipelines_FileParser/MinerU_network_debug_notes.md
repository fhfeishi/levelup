# MinerU App / CLI 网络与模型下载问题排查记录

记录时间：2026-06-03 15:52:18 +08:00

## 背景

本次排查围绕两个相关但不同的问题展开：

1. `pipelines_FileParser/src_mineru.py` 使用 MinerU CLI 解析 PDF 时，模型下载和本地 cache 命中异常。
2. MinerU 桌面客户端在 ClashVergeRev 开启 TUN + 代理模式时显示“未知错误”，关闭 ClashVergeRev 后恢复正常。

机器环境：

- Windows
- Conda 环境：`D:\environment\miniconda\envs\enva`
- MinerU Python 包版本：`mineru-3.2.2`
- MinerU 桌面客户端：`D:\develop\mineru\MinerU.exe`
- Hugging Face cache：`E:\local_models\huggingface\cache`
- ModelScope cache：`E:\local_models\modelscope`
- ClashVergeRev / Mihomo 开启 TUN 后 DNS：`198.18.0.2`

## 一、MinerU CLI 模型源与 cache 问题

### 现象

运行 `src_mineru.py` 时，MinerU 默认下载 Hugging Face 模型失败：

```text
huggingface_hub.errors.FileMetadataError:
Distant resource does not seem to be on huggingface.co.

huggingface_hub.errors.LocalEntryNotFoundError:
cannot find the requested files in the local cache
```

报错涉及的模型包括：

```text
opendatalab/MinerU2.5-Pro-2605-1.2B
opendatalab/MinerU2.5-Pro-2604-1.2B
opendatalab/PDF-Extract-Kit-1.0
```

### 环境变量确认

当前环境变量实际生效：

```powershell
$env:HF_ENDPOINT
# https://hf-mirror.com

$env:HF_HOME
# E:\local_models\huggingface\cache
```

在 `enva` 里 `huggingface_hub` 读取结果：

```text
ENDPOINT=https://hf-mirror.com
HF_HOME=E:\local_models\huggingface\cache
HF_HUB_CACHE=E:\local_models\huggingface\cache\hub
```

注意：正确变量名是 `HF_HOME`，不是 `FH_HOME`。

### cache 状态

Hugging Face cache 里出现了模型目录，但内容不完整：

```text
E:\local_models\huggingface\cache\hub\models--opendatalab--MinerU2.5-Pro-2605-1.2B\refs
```

缺少：

```text
snapshots\<revision>\...
```

所以对 `snapshot_download()` 来说，本地 cache 仍不可用。

### hf-mirror 的关键问题

`HF_ENDPOINT=https://hf-mirror.com` 确实被读取了，但 `hf-mirror.com` 对某些 Hugging Face 文件下载地址返回 308，并重定向回官方站：

```text
https://hf-mirror.com/opendatalab/PDF-Extract-Kit-1.0/resolve/main/...

HTTP/1.1 308 Permanent Redirect
Location: https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/...
```

因此：

- repo 元信息可能能获取；
- 具体文件下载、HEAD 元数据校验可能失败；
- 本地 cache 又不完整时，就报 `LocalEntryNotFoundError`。

### 解决方案

CLI CPU pipeline 当前更稳定的方案是使用 ModelScope：

```python
env.setdefault("MODELSCOPE_CACHE", r"E:\local_models\modelscope")
env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
env.setdefault("MINERU_DEVICE_MODE", "cpu")
```

无 GPU 机器必须避免默认 VLM / hybrid 后端，改用 pipeline：

```python
command = [
    mineru_bin,
    "-p", input_file,
    "-o", output_file,
    "-b", "pipeline",
    "-m", "auto",
]
```

如果要测试 Hugging Face pipeline，建议使用官方 endpoint，而不是 hf-mirror：

```powershell
$env:MINERU_MODEL_SOURCE="huggingface"
$env:HF_ENDPOINT="https://huggingface.co"
D:\environment\miniconda\envs\enva\Scripts\mineru.exe -p "input.pdf" -o "output" -b pipeline -m auto -s 0 -e 0
```

该方式已验证：第一页 PDF 解析成功，模型下载成功。

## 二、无 GPU 的 CUDA 报错

### 现象

无 GPU 机器运行 MinerU 默认后端时报错：

```text
ValueError: CUDA is not available
```

### 原因

MinerU CLI 默认后端是：

```text
hybrid-auto-engine
```

该后端会走 VLM / lmdeploy，本机会尝试初始化 CUDA。

### 修复

改用 CPU 友好的 pipeline 后端：

```powershell
mineru -p "input.pdf" -o "output" -b pipeline -m auto
```

并设置：

```powershell
$env:MINERU_DEVICE_MODE="cpu"
```

## 三、MinerU App “未知错误”真实原因

### 现象

MinerU 桌面客户端在 ClashVergeRev 开启 TUN + 代理模式时显示“未知错误”。

关闭 ClashVergeRev 后恢复正常。

### 日志位置

MinerU App 日志位于：

```text
C:\Users\User\AppData\Roaming\MinerU\logs\main.log
```

### 真实错误

日志显示，MinerU 云端任务其实已经完成：

```text
state: 'done'
full_zip_url: 'https://cdn-mineru.openxlab.org.cn/pdf/.../1986071b-f877-48dd-94d3-adcf21337628.zip'
```

但客户端下载结果 ZIP 失败：

```text
download failed: Error: connect ETIMEDOUT 198.18.0.19:443
```

所以流程是：

```text
PDF 上传成功
MinerU 服务器解析成功
服务器返回结果 ZIP 地址
本机下载 ZIP 失败
App 显示“未知错误”
```

这不是模型下载失败，也不是 MinerU 云端解析失败，而是本机下载解析结果失败。

## 四、ClashVergeRev / Mihomo fake-ip 问题

### 初始状态

开启 TUN 后：

```text
DNS Server: 198.18.0.2
Default route: Mihomo -> 198.18.0.2
System proxy: 127.0.0.1:7897
```

部分域名被分配 fake IP：

```text
cdn-mineru.openxlab.org.cn -> 198.18.0.10
mineru.oss-cn-shanghai.aliyuncs.com -> 198.18.0.14
mirrors.tuna.tsinghua.edu.cn -> 198.18.0.15
huggingface.co -> 198.18.0.x
```

即使执行：

```powershell
nslookup mirrors.tuna.tsinghua.edu.cn 223.5.5.5
```

仍然返回 `198.18.0.x`，说明 TUN 的 DNS hijack 接管了 DNS 请求，并没有真的查询 `223.5.5.5`。

### 错误配置

原配置中使用了：

```yaml
fake-ip-filter:
  - '*.cn'
  - '*.aliyuncs.com'
```

实际表现：

- `*.cn` 不适合匹配多层域名，如 `mirrors.tuna.tsinghua.edu.cn`；
- `*.aliyuncs.com` 不匹配 `mineru.oss-cn-shanghai.aliyuncs.com`；
- 结果这些域名仍然被分配 fake IP。

### 修复配置

将 fake-ip-filter 改成后缀匹配：

```yaml
fake-ip-filter:
  - +.cn
  - +.aliyuncs.com
  - +.openxlab.org.cn
  - +.tsinghua.edu.cn
  - hf-mirror.com
```

修改过的文件：

```text
C:\Users\User\AppData\Roaming\io.github.clash-verge-rev.clash-verge-rev\dns_config.yaml
C:\Users\User\AppData\Roaming\io.github.clash-verge-rev.clash-verge-rev\clash-verge.yaml
```

注意：`dns_config.yaml` 是源配置，`clash-verge.yaml` 是生成配置。只改源配置后，生成配置不一定立刻同步。本次为了验证，两个文件都改了。

### 重启与验证

重启 Mihomo 内核并清 DNS：

```powershell
Get-Process verge-mihomo -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 6
ipconfig /flushdns
```

验证：

```powershell
nslookup mineru.oss-cn-shanghai.aliyuncs.com
nslookup cdn-mineru.openxlab.org.cn
nslookup mirrors.tuna.tsinghua.edu.cn
nslookup hf-mirror.com
```

修复后返回真实 IP：

```text
mineru.oss-cn-shanghai.aliyuncs.com -> 180.163.123.134
cdn-mineru.openxlab.org.cn -> 111.123.55.17 / 119.0.101.213 / ...
mirrors.tuna.tsinghua.edu.cn -> 101.6.15.130
hf-mirror.com -> 160.16.86.14
```

## 五、剩余 CDN TLS 问题

fake-ip 修复后，`cdn-mineru.openxlab.org.cn` 已返回真实 IP，但测试旧 ZIP URL 时仍出现 TLS 握手失败：

```text
curl: (35) schannel: failed to receive handshake, SSL/TLS connection failed
```

同时端口连通性正常：

```text
Test-NetConnection 111.123.55.17 -Port 443
TcpTestSucceeded: True
```

这说明：

- DNS fake-ip 问题已解决；
- 但当前网络到 `cdn-mineru.openxlab.org.cn` 的 TLS 链路可能仍不稳定；
- 可能需要让该 CDN 走代理，而不是 DIRECT。

### 2026-06-03 追加案例：同一 CDN 域名部分边缘 IP 失败

源文件：

```text
D:\ddesktop\FileParser\sources\main_20260602_2.pdf
```

MinerU App 日志显示云端任务完成，但下载结果 ZIP 失败：

```text
full_zip_url: https://cdn-mineru.openxlab.org.cn/pdf/2026-06-03/dfd92a54-2de9-4e86-93f0-3a926950f653.zip
download failed: Error: Client network socket disconnected before secure TLS connection was established
```

此时 DNS 已经不是 fake IP：

```text
cdn-mineru.openxlab.org.cn -> 111.123.55.x / 119.96.89.x / ...
```

但不同 CDN 边缘 IP 的 TLS 表现不同：

```text
111.123.55.19 -> TLS failed
111.123.55.17 -> TLS failed
119.96.89.97 -> TLS failed
119.96.89.69 -> TLS failed
124.239.239.72 -> HTTP/1.1 200 OK
119.0.101.213 -> HTTP/1.1 200 OK
```

可用的临时下载方式：

```powershell
curl.exe -L --http1.1 --max-time 120 `
  --resolve "cdn-mineru.openxlab.org.cn:443:124.239.239.72" `
  "https://cdn-mineru.openxlab.org.cn/pdf/2026-06-03/dfd92a54-2de9-4e86-93f0-3a926950f653.zip" `
  -o "D:\ddesktop\FileParser\output_mineru\dfd92a54-2de9-4e86-93f0-3a926950f653.zip"
```

下载成功后可手动解压到 MinerU App 期望的输出目录：

```powershell
Expand-Archive `
  -LiteralPath "D:\ddesktop\FileParser\output_mineru\dfd92a54-2de9-4e86-93f0-3a926950f653.zip" `
  -DestinationPath "D:\ddesktop\FileParser\output_mineru\main_20260602_2.pdf-425f9fe6-6ae1-4ec7-9285-2896625091d5" `
  -Force
```

经验：DNS 返回真实 IP 不等于问题结束。CDN 解析到的某些边缘节点可能存在 TLS 握手异常，可以用 `curl --resolve` 对多个 IP 逐个验证。

建议 Clash 规则：

```yaml
rules:
  - DOMAIN,cdn-mineru.openxlab.org.cn,🚀 节点选择
  - DOMAIN-SUFFIX,openxlab.org.cn,🚀 节点选择
  - DOMAIN-SUFFIX,aliyuncs.com,DIRECT
```

这些规则要放在 `.cn` / `GEOIP,CN,DIRECT` 之前。

`🚀 节点选择` 替换成当前配置中实际可用的代理组名。

## 六、常用排查命令

### 查看代理与 DNS 状态

```powershell
netsh winhttp show proxy

Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' |
  Select-Object ProxyEnable,ProxyServer,AutoConfigURL,ProxyOverride

Get-DnsClientServerAddress -AddressFamily IPv4 |
  Where-Object {$_.ServerAddresses.Count -gt 0}

Get-NetRoute -DestinationPrefix '0.0.0.0/0' |
  Sort-Object RouteMetric
```

### 查看 Clash / Mihomo 进程

```powershell
Get-Process |
  Where-Object { $_.ProcessName -match 'clash|verge|mihomo' } |
  Select-Object ProcessName,Id,Path
```

### 查看端口监听

```powershell
Get-NetTCPConnection -LocalPort 7897 -State Listen
```

### 检查域名是否仍是 fake IP

```powershell
nslookup cdn-mineru.openxlab.org.cn
nslookup mineru.oss-cn-shanghai.aliyuncs.com
nslookup mirrors.tuna.tsinghua.edu.cn
```

如果返回 `198.18.0.x`，说明仍在 fake-ip。

### 检查 MinerU App 日志

```powershell
Select-String -Path "$env:APPDATA\MinerU\logs\main.log" `
  -Pattern "error|failed|download|timeout|TLS|ETIMEDOUT|zip|full_zip_url" `
  -CaseSensitive:$false |
  Select-Object -Last 80
```

### 检查 Hugging Face 环境

```powershell
$env:HF_ENDPOINT
$env:HF_HOME
D:\environment\miniconda\envs\enva\Scripts\hf.exe env
```

### 测试 Hugging Face pipeline

```powershell
$env:MINERU_MODEL_SOURCE="huggingface"
$env:HF_ENDPOINT="https://huggingface.co"
D:\environment\miniconda\envs\enva\Scripts\mineru.exe `
  -p "input.pdf" `
  -o "output" `
  -b pipeline `
  -m auto `
  -s 0 `
  -e 0
```

### 测试 ModelScope pipeline

```powershell
$env:MINERU_MODEL_SOURCE="modelscope"
$env:MODELSCOPE_CACHE="E:\local_models\modelscope"
$env:MINERU_DEVICE_MODE="cpu"
D:\environment\miniconda\envs\enva\Scripts\mineru.exe `
  -p "input.pdf" `
  -o "output" `
  -b pipeline `
  -m auto
```

## 七、最终结论

本次问题分为三层：

1. MinerU CLI 默认 VLM / hybrid 后端需要 CUDA，无 GPU 机器应使用 `pipeline`。
2. Hugging Face cache 环境变量 `HF_HOME` 正常，但 `hf-mirror.com` 对部分模型文件会 308 跳回官方源，导致 `huggingface_hub` 元数据校验失败。CLI 更推荐 ModelScope 或官方 Hugging Face endpoint。
3. MinerU App 的“未知错误”不是云端解析失败，而是客户端下载结果 ZIP 失败。根因与 Clash TUN / fake-ip / CDN 连接有关。

关键修复：

```yaml
fake-ip-filter:
  - +.cn
  - +.aliyuncs.com
  - +.openxlab.org.cn
  - +.tsinghua.edu.cn
  - hf-mirror.com
```

`*.cn` 这类写法不适合当前场景，`+.cn` 才是更稳的后缀匹配。
