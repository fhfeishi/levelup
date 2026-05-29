@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Windows + Clash Verge Rev network health check.
rem Default check mode is read-only.
rem Apply mode changes conservative current-user Windows proxy / IPv4 preference / DNS cache only.
rem Usage:
rem   win_cvr.cmd
rem   win_cvr.cmd help
rem   win_cvr.cmd check
rem   win_cvr.cmd 7890
rem   win_cvr.cmd apply
rem   win_cvr.cmd apply 7890

if /I "%~1"=="help" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage

set "MODE=check"
set "PORT=7897"
if /I "%~1"=="check" (
  set "MODE=check"
  if not "%~2"=="" set "PORT=%~2"
) else if /I "%~1"=="apply" (
  set "MODE=apply"
  if not "%~2"=="" set "PORT=%~2"
) else if not "%~1"=="" (
  set "PORT=%~1"
)
set "TIMEOUT=10"
set "LOG=%~dp0win_cvr_report_%COMPUTERNAME%_%DATE:/=-%_%TIME::=-%.log"
set "LOG=%LOG: =_%"

set /a CURL4_OK=0
set /a CURL4_FAIL=0
set /a CURL6_OK=0
set /a CURL6_FAIL=0
set /a CURL_OK=0
set /a CURL_FAIL=0
set /a PROXY_OK=0
set /a PROXY_FAIL=0

echo Windows + Clash Verge Rev network health check
echo Log: "%LOG%"
echo Mode: %MODE%
echo Port: %PORT%
echo Timeout: %TIMEOUT%s
echo.

> "%LOG%" echo Windows + Clash Verge Rev network health check
>>"%LOG%" echo Started: %DATE% %TIME%
>>"%LOG%" echo Computer: %COMPUTERNAME%
>>"%LOG%" echo User: %USERNAME%
>>"%LOG%" echo Mode: %MODE%
>>"%LOG%" echo Port: %PORT%
>>"%LOG%" echo Timeout: %TIMEOUT%s

if /I "%MODE%"=="apply" call :apply_config

call :section "0. Basic Windows network state"
call :run "ver" "ver"
call :run "hostname" "hostname"
call :run "ipconfig /all" "ipconfig /all"
call :run "route print" "route print"
call :run "netsh interface ipv4 show interfaces" "netsh interface ipv4 show interfaces"
call :run "netsh interface ipv6 show interfaces" "netsh interface ipv6 show interfaces"
call :run "netsh interface ipv6 show prefixpolicies" "netsh interface ipv6 show prefixpolicies"

call :section "1. Windows proxy and Clash port state"
echo.
echo --- Current process proxy env
>>"%LOG%" echo.
>>"%LOG%" echo --- Current process proxy env
set | findstr /I PROXY >>"%LOG%" 2>&1
if errorlevel 1 (echo INFO: no process proxy env found) else (echo OK: process proxy env checked)

echo.
echo --- WinINET proxy registry
>>"%LOG%" echo.
>>"%LOG%" echo --- WinINET proxy registry
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyEnable >>"%LOG%" 2>&1
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer >>"%LOG%" 2>&1
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v AutoConfigURL >>"%LOG%" 2>&1
echo OK: WinINET proxy registry queried

echo.
echo --- WinHTTP proxy
>>"%LOG%" echo.
>>"%LOG%" echo --- WinHTTP proxy
netsh winhttp show proxy >>"%LOG%" 2>&1
if errorlevel 1 (echo FAIL: WinHTTP proxy) else (echo OK: WinHTTP proxy)

echo.
echo --- Listening ports around Clash mixed-port
>>"%LOG%" echo.
>>"%LOG%" echo --- Listening ports around Clash mixed-port
netstat -ano | findstr :%PORT% >>"%LOG%" 2>&1
if errorlevel 1 (echo FAIL: no listener found around port %PORT%) else (echo OK: port %PORT% found in netstat)

call :section "2. DNS check with nslookup"
for %%H in (
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
) do call :dns "%%~H"

call :section "3. IPv4 / IPv6 HTTPS check"
for %%U in (
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
) do (
  call :curlip 4 "%%~U"
  call :curlip 6 "%%~U"
)

call :section "4. Default HTTPS check"
for %%U in (
  https://www.microsoft.com/
  https://www.baidu.com/
  https://www.zhihu.com/
  https://www.bing.com/
  https://chatgpt.com/
  https://github.com/
  https://pypi.org/simple/pillow/
  https://mirrors.aliyun.com/pypi/simple/
  https://developer.download.nvidia.com/compute/cuda/repos/
) do call :curldefault "%%~U"

call :section "5. Explicit Clash proxy check"
for %%U in (
  https://github.com/
  https://google.com/
  https://pypi.org/simple/pillow/
  https://files.pythonhosted.org/
  https://developer.download.nvidia.com/compute/cuda/repos/
  https://astral.sh/
) do (
  echo.
  echo --- curl proxy http://127.0.0.1:%PORT% %%~U
  >>"%LOG%" echo.
  >>"%LOG%" echo --- curl proxy http://127.0.0.1:%PORT% %%~U
  curl.exe -I -L --max-time %TIMEOUT% --proxy "http://127.0.0.1:%PORT%" "%%~U" >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo FAIL: proxy %%~U
    >>"%LOG%" echo RESULT: FAIL
    set /a PROXY_FAIL+=1
  ) else (
    echo OK: proxy %%~U
    >>"%LOG%" echo RESULT: OK
    set /a PROXY_OK+=1
  )
)

call :section "6. Git / Python / Node tool state on Windows"
echo.
echo --- git
>>"%LOG%" echo.
>>"%LOG%" echo --- git
where git >>"%LOG%" 2>&1
if errorlevel 1 echo WARN: git not found
git config --global --get-regexp http.*proxy >>"%LOG%" 2>&1
if errorlevel 1 echo INFO: no global git proxy configured
git ls-remote https://github.com/git/git.git HEAD >>"%LOG%" 2>&1
if errorlevel 1 echo FAIL: git ls-remote GitHub
if not errorlevel 1 echo OK: git ls-remote GitHub

echo.
echo --- python / pip
>>"%LOG%" echo.
>>"%LOG%" echo --- python / pip
where python >>"%LOG%" 2>&1
if errorlevel 1 echo WARN: python not found
python -m pip config list -v >>"%LOG%" 2>&1
python -m pip index versions pip --timeout 10 >>"%LOG%" 2>&1
if errorlevel 1 echo FAIL: pip index query
if not errorlevel 1 echo OK: pip index query

echo.
echo --- uv
>>"%LOG%" echo.
>>"%LOG%" echo --- uv
where uv >>"%LOG%" 2>&1
if errorlevel 1 echo WARN: uv not found
uv --version >>"%LOG%" 2>&1
set "PYEXE="
py -3 -c "import sys; print(sys.executable)" > "%TEMP%\win_cvr_pyexe.txt" 2>>"%LOG%"
if exist "%TEMP%\win_cvr_pyexe.txt" set /p PYEXE=<"%TEMP%\win_cvr_pyexe.txt"
if exist "%TEMP%\win_cvr_pyexe.txt" del "%TEMP%\win_cvr_pyexe.txt" >nul 2>nul
if not "%PYEXE%"=="" (
  echo Python for uv: %PYEXE%>>"%LOG%"
  uv pip install --dry-run --python "%PYEXE%" --index-url https://pypi.org/simple pip >>"%LOG%" 2>&1
) else (
  echo WARN: py -3 did not return a Python executable for uv>>"%LOG%"
  uv pip install --dry-run --index-url https://pypi.org/simple pip >>"%LOG%" 2>&1
)
if errorlevel 1 echo FAIL: uv dry-run pip network check
if not errorlevel 1 echo OK: uv dry-run pip network check

echo.
echo --- node / npm
>>"%LOG%" echo.
>>"%LOG%" echo --- node / npm
where node >>"%LOG%" 2>&1
where npm >>"%LOG%" 2>&1
if errorlevel 1 echo WARN: npm not found
call npm config get proxy >>"%LOG%" 2>&1
call npm config get https-proxy >>"%LOG%" 2>&1
call npm view npm version --registry=https://registry.npmjs.org/ >>"%LOG%" 2>&1
if errorlevel 1 echo FAIL: npm registry query
if not errorlevel 1 echo OK: npm registry query

call :section "7. Summary"
echo IPv4 curl: !CURL4_OK! ok, !CURL4_FAIL! failed
echo IPv6 curl: !CURL6_OK! ok, !CURL6_FAIL! failed
echo Default curl: !CURL_OK! ok, !CURL_FAIL! failed
echo Explicit proxy curl: !PROXY_OK! ok, !PROXY_FAIL! failed
echo.
echo Interpretation:
echo - If IPv4 mostly passes and IPv6 mostly fails, prefer IPv4 on this machine.
echo - If default curl fails but explicit proxy passes, use 127.0.0.1:%PORT% as a tool-level fallback.
echo - If DNS returns 198.18.x.x, Clash fake-ip DNS is involved; this is normal for proxied domains.
echo - If domestic mirrors return 198.18.x.x, check Clash fake-ip-filter / DIRECT rules.
echo - If the Clash port is not listening, System Proxy pointing to that port will break tools and browsers.
echo.
echo Clash-side suggestions:
if !CURL6_FAIL! GTR !CURL6_OK! (
  echo - IPv6 looks unreliable on this machine. In Clash, prefer: ipv6: false
  echo - In Clash DNS, prefer: ipv6: false
) else (
  echo - IPv6 did not look broadly worse than IPv4 in this run. Keep IPv6 only if real browsing/downloads are stable.
)
echo - If using fake-ip, add domestic/dev mirrors to fake-ip-filter:
echo   *.aliyun.com
echo   *.tsinghua.edu.cn
echo   *.tuna.tsinghua.edu.cn
echo   mirrors.tuna.tsinghua.edu.cn
echo   pypi.tuna.tsinghua.edu.cn
echo   pypi.org
echo   *.pypi.org
echo   files.pythonhosted.org
echo   *.pythonhosted.org
echo   github.com
echo   *.github.com
echo   *.githubusercontent.com
echo   raw.githubusercontent.com
echo   objects.githubusercontent.com
echo   nodejs.org
echo   registry.npmjs.org
echo   developer.download.nvidia.com
echo   astral.sh
echo.
echo Full log saved to:
echo "%LOG%"

>>"%LOG%" echo.
>>"%LOG%" echo ==== Summary ====
>>"%LOG%" echo IPv4 curl: !CURL4_OK! ok, !CURL4_FAIL! failed
>>"%LOG%" echo IPv6 curl: !CURL6_OK! ok, !CURL6_FAIL! failed
>>"%LOG%" echo Default curl: !CURL_OK! ok, !CURL_FAIL! failed
>>"%LOG%" echo Explicit proxy curl: !PROXY_OK! ok, !PROXY_FAIL! failed
>>"%LOG%" echo.
>>"%LOG%" echo Clash-side suggestions:
if !CURL6_FAIL! GTR !CURL6_OK! (
  >>"%LOG%" echo - IPv6 looks unreliable on this machine. In Clash, prefer: ipv6: false
  >>"%LOG%" echo - In Clash DNS, prefer: ipv6: false
) else (
  >>"%LOG%" echo - IPv6 did not look broadly worse than IPv4 in this run.
)
>>"%LOG%" echo - fake-ip-filter candidates: *.aliyun.com, *.tsinghua.edu.cn, *.tuna.tsinghua.edu.cn, mirrors.tuna.tsinghua.edu.cn, pypi.tuna.tsinghua.edu.cn, pypi.org, *.pypi.org, files.pythonhosted.org, *.pythonhosted.org, github.com, *.github.com, *.githubusercontent.com, raw.githubusercontent.com, objects.githubusercontent.com, nodejs.org, registry.npmjs.org, developer.download.nvidia.com, astral.sh
>>"%LOG%" echo Finished: %DATE% %TIME%

exit /b 0

:usage
echo Windows + Clash Verge Rev helper
echo.
echo Usage:
echo   win_cvr.cmd
echo   win_cvr.cmd check
echo   win_cvr.cmd check 7897
echo   win_cvr.cmd 7897
echo   win_cvr.cmd apply
echo   win_cvr.cmd apply 7897
echo.
echo Modes:
echo   check   Read-only network health check. This is the default.
echo   apply   Conservative Windows setup, then run check.
echo.
echo Default port:
echo   7897
echo.
echo Logs:
echo   The script writes a detailed log beside itself:
echo   %~dp0win_cvr_report_*.log
echo.
echo Apply mode changes:
echo   - Enable current-user Windows System Proxy.
echo   - Set System Proxy to 127.0.0.1:^<port^>.
echo   - Prefer IPv4 using Windows IPv6 prefix policy.
echo   - Flush Windows DNS cache.
echo.
echo Apply mode does NOT:
echo   - Disable network adapter IPv6.
echo   - Change WinHTTP proxy.
echo   - Edit Clash profiles.
echo   - Reset network stack.
echo.
echo Examples:
echo   win_cvr.cmd check 7897
echo   win_cvr.cmd apply 7897
echo.
exit /b 0

:section
echo.
echo ============================================================
echo %~1
echo ============================================================
>>"%LOG%" echo.
>>"%LOG%" echo ============================================================
>>"%LOG%" echo %~1
>>"%LOG%" echo ============================================================
exit /b 0

:apply_config
call :section "A. Apply conservative Windows network configuration"
echo This mode changes only conservative Windows settings:
echo - Enable WinINET/System Proxy for current user: 127.0.0.1:%PORT%
echo - Prefer IPv4 over IPv6 using Windows prefix policy
echo - Flush Windows DNS cache
echo It does NOT disable any network adapter IPv6 binding.
echo It does NOT edit Clash profiles.
echo.
>>"%LOG%" echo Apply mode changes only conservative Windows settings.
>>"%LOG%" echo It does NOT disable any network adapter IPv6 binding.
>>"%LOG%" echo It does NOT edit Clash profiles.

echo --- Admin check
>>"%LOG%" echo.
>>"%LOG%" echo --- Admin check
net session >nul 2>&1
if errorlevel 1 (
  echo WARN: not running as Administrator. IPv4 preference may fail.
  >>"%LOG%" echo WARN: not running as Administrator. IPv4 preference may fail.
) else (
  echo OK: running as Administrator.
  >>"%LOG%" echo OK: running as Administrator.
)

echo.
echo --- Enable current-user System Proxy
>>"%LOG%" echo.
>>"%LOG%" echo --- Enable current-user System Proxy
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyEnable /t REG_DWORD /d 1 /f >>"%LOG%" 2>&1
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer /t REG_SZ /d "127.0.0.1:%PORT%" /f >>"%LOG%" 2>&1
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyOverride /t REG_SZ /d "localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;192.168.*;<local>" /f >>"%LOG%" 2>&1
echo OK: System Proxy set to 127.0.0.1:%PORT% for current user.

echo.
echo --- Prefer IPv4 prefix policy
>>"%LOG%" echo.
>>"%LOG%" echo --- Prefer IPv4 prefix policy
netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4 >>"%LOG%" 2>&1
if errorlevel 1 (
  echo WARN: failed to set IPv4 prefix policy. Try running CMD as Administrator.
  >>"%LOG%" echo RESULT: WARN failed to set IPv4 prefix policy.
) else (
  echo OK: IPv4 prefix policy raised.
  >>"%LOG%" echo RESULT: OK IPv4 prefix policy raised.
)

echo.
echo --- Flush DNS cache
>>"%LOG%" echo.
>>"%LOG%" echo --- Flush DNS cache
ipconfig /flushdns >>"%LOG%" 2>&1
if errorlevel 1 (
  echo WARN: DNS flush failed.
  >>"%LOG%" echo RESULT: WARN DNS flush failed.
) else (
  echo OK: DNS cache flushed.
  >>"%LOG%" echo RESULT: OK DNS cache flushed.
)

echo.
echo --- WinHTTP proxy left unchanged
echo INFO: WinHTTP is not changed by this script. If services need proxy, configure it deliberately.
>>"%LOG%" echo.
>>"%LOG%" echo --- WinHTTP proxy left unchanged
>>"%LOG%" echo INFO: WinHTTP is not changed by this script.
exit /b 0

:run
set "TITLE=%~1"
set "CMDLINE=%~2"
echo.
echo --- %TITLE%
>>"%LOG%" echo.
>>"%LOG%" echo --- %TITLE%
cmd /c %CMDLINE% >>"%LOG%" 2>&1
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="0" echo OK: %TITLE%
if "%RUN_RC%"=="0" >>"%LOG%" echo RESULT: OK
if "%RUN_RC%"=="0" exit /b 0
echo FAIL: %TITLE%
>>"%LOG%" echo RESULT: FAIL
exit /b 0

:dns
set "HOST=%~1"
echo.
echo --- nslookup %HOST%
>>"%LOG%" echo.
>>"%LOG%" echo --- nslookup %HOST%
nslookup "%HOST%" >>"%LOG%" 2>&1
if errorlevel 1 (
  echo FAIL: nslookup %HOST%
  >>"%LOG%" echo RESULT: FAIL
) else (
  echo OK: nslookup %HOST%
  >>"%LOG%" echo RESULT: OK
)
exit /b 0

:curlip
set "IPVER=%~1"
set "URL=%~2"
echo.
echo --- curl IPv%IPVER% %URL%
>>"%LOG%" echo.
>>"%LOG%" echo --- curl IPv%IPVER% %URL%
curl.exe -%IPVER% -I -L --max-time %TIMEOUT% "%URL%" >>"%LOG%" 2>&1
if errorlevel 1 (
  echo FAIL: IPv%IPVER% %URL%
  >>"%LOG%" echo RESULT: FAIL
  if "%IPVER%"=="4" set /a CURL4_FAIL+=1
  if "%IPVER%"=="6" set /a CURL6_FAIL+=1
) else (
  echo OK: IPv%IPVER% %URL%
  >>"%LOG%" echo RESULT: OK
  if "%IPVER%"=="4" set /a CURL4_OK+=1
  if "%IPVER%"=="6" set /a CURL6_OK+=1
)
exit /b 0

:curldefault
set "URL=%~1"
echo.
echo --- curl default %URL%
>>"%LOG%" echo.
>>"%LOG%" echo --- curl default %URL%
curl.exe -I -L --max-time %TIMEOUT% "%URL%" >>"%LOG%" 2>&1
if errorlevel 1 (
  echo FAIL: default %URL%
  >>"%LOG%" echo RESULT: FAIL
  set /a CURL_FAIL+=1
) else (
  echo OK: default %URL%
  >>"%LOG%" echo RESULT: OK
  set /a CURL_OK+=1
)
exit /b 0

:curlproxy
set "URL=%~1"
echo.
echo --- curl proxy http://127.0.0.1:%PORT% %URL%
>>"%LOG%" echo.
>>"%LOG%" echo --- curl proxy http://127.0.0.1:%PORT% %URL%
curl.exe -I -L --max-time %TIMEOUT% --proxy "http://127.0.0.1:%PORT%" "%URL%" >>"%LOG%" 2>&1
if errorlevel 1 (
  echo FAIL: proxy %URL%
  >>"%LOG%" echo RESULT: FAIL
  set /a PROXY_FAIL+=1
) else (
  echo OK: proxy %URL%
  >>"%LOG%" echo RESULT: OK
  set /a PROXY_OK+=1
)
exit /b 0
