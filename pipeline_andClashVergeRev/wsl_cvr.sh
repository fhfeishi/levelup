#!/usr/bin/env bash
set -u

# WSL Ubuntu + Windows Clash Verge Rev health check / conservative setup.
# Default check mode is read-only.
# Apply mode changes conservative WSL user/tool settings only:
#   - add proxy helper functions to ~/.bashrc
#   - set git http/https proxy to http://127.0.0.1:<port>
#   - prefer IPv4 in /etc/gai.conf
#   - add apt ForceIPv4 config
# It does not lock /etc/resolv.conf and does not edit Clash profiles.
#
# Usage:
#   bash wsl_cvr.sh
#   bash wsl_cvr.sh help
#   bash wsl_cvr.sh check
#   bash wsl_cvr.sh check 7897
#   bash wsl_cvr.sh apply
#   bash wsl_cvr.sh apply 7897

show_usage() {
  cat <<'EOF'
WSL Ubuntu + Windows Clash Verge Rev helper

Usage:
  bash wsl_cvr.sh
  bash wsl_cvr.sh check
  bash wsl_cvr.sh check 7897
  bash wsl_cvr.sh 7897
  bash wsl_cvr.sh apply
  bash wsl_cvr.sh apply 7897

Modes:
  check   Read-only WSL Ubuntu network health check. This is the default.
  apply   Conservative WSL setup, then run check.

Default port:
  7897

Logs:
  The script writes a detailed log beside itself:
  ./wsl_cvr_report_*.log

Check mode tests:
  - WSL DNS, routes, proxy env, /etc/resolv.conf, /etc/gai.conf
  - IPv4 / IPv6 access to PyPI, GitHub, Google, NVIDIA, npm, Docker, etc.
  - Direct curl and explicit Clash proxy curl
  - git, apt, pip, uv, npm

Apply mode changes:
  - Add proxy_on/proxy_off/proxy_run/proxysudo helpers to ~/.bashrc.
  - Set git http.proxy and https.proxy to http://127.0.0.1:<port>.
  - Prefer IPv4 in /etc/gai.conf.
  - Add apt Acquire::ForceIPv4 config.

Apply mode does NOT:
  - Auto-enable global proxy.
  - Lock or overwrite /etc/resolv.conf.
  - Edit Windows settings.
  - Edit Clash profiles.

Examples:
  bash wsl_cvr.sh check 7897
  bash wsl_cvr.sh apply 7897
EOF
}

case "${1:-}" in
  help|-h|--help)
    show_usage
    exit 0
    ;;
esac

MODE="check"
PORT="7897"
TIMEOUT="10"

case "${1:-}" in
  check|"")
    MODE="check"
    [ "${2:-}" != "" ] && PORT="$2"
    ;;
  apply)
    MODE="apply"
    [ "${2:-}" != "" ] && PORT="$2"
    ;;
  *)
    PORT="$1"
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/wsl_cvr_report_$(hostname)_$(date +%Y%m%d_%H%M%S).log"

CURL4_OK=0
CURL4_FAIL=0
CURL6_OK=0
CURL6_FAIL=0
CURL_OK=0
CURL_FAIL=0
PROXY_OK=0
PROXY_FAIL=0

HOSTS=(
  pypi.org
  files.pythonhosted.org
  mirrors.aliyun.com
  pypi.tuna.tsinghua.edu.cn
  mirrors.tuna.tsinghua.edu.cn
  github.com
  raw.githubusercontent.com
  objects.githubusercontent.com
  google.com
  developer.download.nvidia.com
  astral.sh
  nodejs.org
  registry.npmjs.org
  docker.io
  ghcr.io
)

URLS=(
  https://pypi.org/simple/pillow/
  https://files.pythonhosted.org/
  https://mirrors.aliyun.com/pypi/simple/
  https://pypi.tuna.tsinghua.edu.cn/simple/
  https://mirrors.tuna.tsinghua.edu.cn/
  https://github.com/
  https://raw.githubusercontent.com/
  https://objects.githubusercontent.com/
  https://google.com/
  https://developer.download.nvidia.com/compute/cuda/repos/
  https://astral.sh/
  https://nodejs.org/dist/
  https://registry.npmjs.org/
  https://hub.docker.com/
  https://ghcr.io/
)

DEFAULT_URLS=(
  https://github.com/
  https://pypi.org/simple/pillow/
  https://mirrors.aliyun.com/pypi/simple/
  https://mirrors.tuna.tsinghua.edu.cn/
  https://google.com/
  https://developer.download.nvidia.com/compute/cuda/repos/
  https://astral.sh/
)

PROXY_URLS=(
  https://github.com/
  https://google.com/
  https://pypi.org/simple/pillow/
  https://files.pythonhosted.org/
  https://developer.download.nvidia.com/compute/cuda/repos/
  https://astral.sh/
)

say() {
  echo "$*"
  echo "$*" >>"$LOG"
}

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
  {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
  } >>"$LOG"
}

run() {
  local title="$1"
  shift
  echo
  echo "--- $title"
  {
    echo
    echo "--- $title"
    "$@"
  } >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK: $title"
    echo "RESULT: OK" >>"$LOG"
  else
    echo "FAIL: $title"
    echo "RESULT: FAIL rc=$rc" >>"$LOG"
  fi
  return 0
}

curl_ip() {
  local ipver="$1"
  local url="$2"
  echo
  echo "--- curl IPv${ipver} $url"
  {
    echo
    echo "--- curl IPv${ipver} $url"
    curl "-${ipver}" -I -L --max-time "$TIMEOUT" "$url"
  } >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK: IPv${ipver} $url"
    echo "RESULT: OK" >>"$LOG"
    if [ "$ipver" = "4" ]; then CURL4_OK=$((CURL4_OK + 1)); else CURL6_OK=$((CURL6_OK + 1)); fi
  else
    echo "FAIL: IPv${ipver} $url"
    echo "RESULT: FAIL rc=$rc" >>"$LOG"
    if [ "$ipver" = "4" ]; then CURL4_FAIL=$((CURL4_FAIL + 1)); else CURL6_FAIL=$((CURL6_FAIL + 1)); fi
  fi
}

curl_default() {
  local url="$1"
  echo
  echo "--- curl default $url"
  {
    echo
    echo "--- curl default $url"
    curl -I -L --max-time "$TIMEOUT" "$url"
  } >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK: default $url"
    echo "RESULT: OK" >>"$LOG"
    CURL_OK=$((CURL_OK + 1))
  else
    echo "FAIL: default $url"
    echo "RESULT: FAIL rc=$rc" >>"$LOG"
    CURL_FAIL=$((CURL_FAIL + 1))
  fi
}

curl_proxy() {
  local url="$1"
  echo
  echo "--- curl proxy http://127.0.0.1:${PORT} $url"
  {
    echo
    echo "--- curl proxy http://127.0.0.1:${PORT} $url"
    curl -I -L --max-time "$TIMEOUT" --proxy "http://127.0.0.1:${PORT}" "$url"
  } >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK: proxy $url"
    echo "RESULT: OK" >>"$LOG"
    PROXY_OK=$((PROXY_OK + 1))
  else
    echo "FAIL: proxy $url"
    echo "RESULT: FAIL rc=$rc" >>"$LOG"
    PROXY_FAIL=$((PROXY_FAIL + 1))
  fi
}

need_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "WARN: sudo not found; cannot run: $*" | tee -a "$LOG"
    return 1
  fi
}

apply_config() {
  section "A. Apply conservative WSL Ubuntu configuration"
  say "This mode changes conservative WSL settings:"
  say "- Add proxy helper functions to ~/.bashrc, without auto-enabling proxy."
  say "- Configure git http.proxy and https.proxy to http://127.0.0.1:${PORT}."
  say "- Prefer IPv4 using /etc/gai.conf."
  say "- Configure apt Acquire::ForceIPv4."
  say "It does NOT lock /etc/resolv.conf."
  say "It does NOT edit Clash profiles."

  local bashrc="${HOME}/.bashrc"
  local backup="${bashrc}.wsl_cvr.bak.$(date +%Y%m%d_%H%M%S)"
  if [ -f "$bashrc" ]; then
    cp "$bashrc" "$backup"
    say "Backed up ~/.bashrc to $backup"
  fi

  if grep -q "WSL Ubuntu -> Windows Clash" "$bashrc" 2>/dev/null; then
    say "Proxy helper block already exists in ~/.bashrc; leaving it in place."
  else
    cat >>"$bashrc" <<EOF

# ================================================================
# WSL Ubuntu -> Windows Clash
# Added by wsl_cvr.sh. This does not auto-enable proxy.
# ================================================================
_PROXY_HOST="127.0.0.1"
_PROXY_PORT="${PORT}"
_PROXY_HTTP="http://\${_PROXY_HOST}:\${_PROXY_PORT}"
_PROXY_SOCKS="socks5://\${_PROXY_HOST}:\${_PROXY_PORT}"
_NO_PROXY="localhost,127.0.0.1,::1,.local,10.*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,192.168.*"

proxy_on() {
  export http_proxy="\${_PROXY_HTTP}"
  export https_proxy="\${_PROXY_HTTP}"
  export all_proxy="\${_PROXY_SOCKS}"
  export HTTP_PROXY="\${_PROXY_HTTP}"
  export HTTPS_PROXY="\${_PROXY_HTTP}"
  export ALL_PROXY="\${_PROXY_SOCKS}"
  export no_proxy="\${_NO_PROXY}"
  export NO_PROXY="\${_NO_PROXY}"
  echo "Proxy ON -> \${_PROXY_HTTP}"
}

proxy_off() {
  unset http_proxy https_proxy all_proxy no_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  echo "Proxy OFF"
}

proxy_status() {
  if [ -n "\${http_proxy:-}" ]; then
    echo "Proxy ON -> \${http_proxy}"
  else
    echo "Proxy OFF"
  fi
}

proxy_run() {
  http_proxy="\${_PROXY_HTTP}" \\
  https_proxy="\${_PROXY_HTTP}" \\
  all_proxy="\${_PROXY_SOCKS}" \\
  HTTP_PROXY="\${_PROXY_HTTP}" \\
  HTTPS_PROXY="\${_PROXY_HTTP}" \\
  ALL_PROXY="\${_PROXY_SOCKS}" \\
  "\$@"
}

proxysudo() {
  sudo -E env \\
  http_proxy="\${_PROXY_HTTP}" \\
  https_proxy="\${_PROXY_HTTP}" \\
  all_proxy="\${_PROXY_SOCKS}" \\
  "\$@"
}
EOF
    say "Added proxy helper functions to ~/.bashrc."
  fi

  if command -v git >/dev/null 2>&1; then
    git config --global http.proxy "http://127.0.0.1:${PORT}"
    git config --global https.proxy "http://127.0.0.1:${PORT}"
    say "Configured git proxy to http://127.0.0.1:${PORT}."
  else
    say "WARN: git not found; skipped git proxy config."
  fi

  if [ -f /etc/gai.conf ]; then
    need_sudo cp /etc/gai.conf "/etc/gai.conf.wsl_cvr.bak.$(date +%Y%m%d_%H%M%S)" || true
  fi
  if grep -qE '^[[:space:]]*precedence[[:space:]]+::ffff:0:0/96[[:space:]]+100' /etc/gai.conf 2>/dev/null; then
    say "/etc/gai.conf already prefers IPv4."
  else
    echo 'precedence ::ffff:0:0/96  100' | need_sudo tee -a /etc/gai.conf >/dev/null
    say "Added IPv4 precedence to /etc/gai.conf."
  fi

  echo 'Acquire::ForceIPv4 "true";' | need_sudo tee /etc/apt/apt.conf.d/99force-ipv4 >/dev/null
  say "Configured apt ForceIPv4 at /etc/apt/apt.conf.d/99force-ipv4."

  say "Apply complete. Open a new shell or run: source ~/.bashrc"
}

summary() {
  section "7. Summary"
  say "IPv4 curl: ${CURL4_OK} ok, ${CURL4_FAIL} failed"
  say "IPv6 curl: ${CURL6_OK} ok, ${CURL6_FAIL} failed"
  say "Default curl: ${CURL_OK} ok, ${CURL_FAIL} failed"
  say "Explicit proxy curl: ${PROXY_OK} ok, ${PROXY_FAIL} failed"
  say ""
  say "Interpretation:"
  say "- If IPv4 mostly passes and IPv6 mostly fails, prefer IPv4 in WSL."
  say "- If default curl fails but explicit proxy passes, use proxy_run/proxysudo or tool-level proxy."
  say "- If DNS returns 198.18.x.x, Clash fake-ip DNS is involved; this is normal for proxied domains."
  say "- If domestic mirrors return 198.18.x.x, check Clash fake-ip-filter / DIRECT rules."
  say "- If apt normal update fails but ForceIPv4 works, keep apt ForceIPv4."
  say ""
  say "Recommended WSL-side config:"
  if [ "$CURL6_FAIL" -gt "$CURL6_OK" ]; then
    say "- IPv6 looks unreliable. Keep /etc/gai.conf IPv4 precedence."
    say "- In Clash, prefer ipv6: false and dns.ipv6: false."
  else
    say "- IPv6 did not look broadly worse than IPv4. Keep it only if real downloads are stable."
  fi
  say "- Do not auto-enable global proxy unless direct mode fails."
  say "- Keep proxy_on/proxy_run/proxysudo as emergency switches."
  say "- For git, http.proxy/https.proxy can point to http://127.0.0.1:${PORT}."
  say ""
  say "Clash fake-ip-filter candidates:"
  say "  *.aliyun.com"
  say "  *.tsinghua.edu.cn"
  say "  *.tuna.tsinghua.edu.cn"
  say "  mirrors.tuna.tsinghua.edu.cn"
  say "  pypi.tuna.tsinghua.edu.cn"
  say "  pypi.org"
  say "  *.pypi.org"
  say "  files.pythonhosted.org"
  say "  *.pythonhosted.org"
  say "  github.com"
  say "  *.github.com"
  say "  *.githubusercontent.com"
  say "  raw.githubusercontent.com"
  say "  objects.githubusercontent.com"
  say "  nodejs.org"
  say "  registry.npmjs.org"
  say "  developer.download.nvidia.com"
  say "  astral.sh"
  say ""
  say "Full log saved to: $LOG"
}

echo "WSL Ubuntu + Clash Verge Rev network health check"
echo "Log: $LOG"
echo "Mode: $MODE"
echo "Port: $PORT"
echo "Timeout: ${TIMEOUT}s"
echo
{
  echo "WSL Ubuntu + Clash Verge Rev network health check"
  echo "Started: $(date)"
  echo "Host: $(hostname)"
  echo "User: ${USER:-unknown}"
  echo "Mode: $MODE"
  echo "Port: $PORT"
  echo "Timeout: ${TIMEOUT}s"
} >"$LOG"

if [ "$MODE" = "apply" ]; then
  apply_config
fi

section "0. Basic WSL Ubuntu state"
run "uname" uname -a
run "os-release" cat /etc/os-release
run "wsl version if available" sh -lc 'command -v wsl.exe >/dev/null 2>&1 && wsl.exe --version || true'
run "proxy env" sh -lc 'env | grep -i _proxy || true'
run "resolv.conf" cat /etc/resolv.conf
run "routes" ip route
run "IPv6 routes" sh -lc 'ip -6 route || true'
run "gai IPv4 precedence" sh -lc "grep -nE '^[[:space:]]*precedence[[:space:]]+::ffff:0:0/96' /etc/gai.conf || true"

section "1. Windows Clash port reachability from WSL"
run "curl Clash port localhost" curl -I --max-time "$TIMEOUT" --proxy "http://127.0.0.1:${PORT}" https://pypi.org/simple/pillow/
run "Windows host from resolv.conf" sh -lc 'awk "/^nameserver / {print \$2; exit}" /etc/resolv.conf'

section "2. DNS check with getent"
for h in "${HOSTS[@]}"; do
  echo
  echo "--- getent $h"
  {
    echo
    echo "--- getent $h"
    getent ahosts "$h" | head
  } >>"$LOG" 2>&1
  if getent ahosts "$h" >/dev/null 2>&1; then
    echo "OK: getent $h"
    echo "RESULT: OK" >>"$LOG"
  else
    echo "FAIL: getent $h"
    echo "RESULT: FAIL" >>"$LOG"
  fi
done

section "3. IPv4 / IPv6 HTTPS check"
for u in "${URLS[@]}"; do
  curl_ip 4 "$u"
  curl_ip 6 "$u"
done

section "4. Default HTTPS check"
for u in "${DEFAULT_URLS[@]}"; do
  curl_default "$u"
done

section "5. Explicit Clash proxy check"
for u in "${PROXY_URLS[@]}"; do
  curl_proxy "$u"
done

section "6. WSL development tool checks"
run "git proxy config" sh -lc "git config --global --get-regexp 'http.*proxy|https.*proxy' || true"
run "git ls-remote GitHub" git ls-remote https://github.com/git/git.git HEAD
run "apt config ForceIPv4" sh -lc "grep -R 'Acquire::ForceIPv4' /etc/apt/apt.conf.d 2>/dev/null || true"
run "apt update normal" sh -lc "sudo -n true 2>/dev/null && sudo apt-get update || echo 'SKIP: sudo password required for apt update'"
run "apt update ForceIPv4" sh -lc "sudo -n true 2>/dev/null && sudo apt-get -o Acquire::ForceIPv4=true update || echo 'SKIP: sudo password required for apt ForceIPv4 update'"
run "python / pip config" sh -lc "command -v python3 >/dev/null 2>&1 && python3 -m pip config list -v 2>/dev/null || true"
run "pip index query" sh -lc "command -v python3 >/dev/null 2>&1 && python3 -m pip index versions pip --timeout 10 || true"
run "uv dry-run network check" sh -lc 'command -v uv >/dev/null 2>&1 && uv --version && uv pip install --dry-run --python "$(command -v python3)" --index-url https://pypi.org/simple pip || true'
run "npm registry query" sh -lc "command -v npm >/dev/null 2>&1 && npm config get proxy && npm config get https-proxy && npm view npm version --registry=https://registry.npmjs.org/ || true"
run "optional cargo/go checks" sh -lc "command -v cargo >/dev/null 2>&1 && cargo search serde --limit 1 || true; command -v go >/dev/null 2>&1 && GOPROXY=https://proxy.golang.org,direct go env GOPROXY || true"

summary
