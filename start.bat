@echo off
rem ===========================================================================
rem  llama.cpp launcher - fallback start script
rem
rem  Goal: start the GUI with a *windowed* Python so that no console window
rem  appears at any point.  pyw.exe and pythonw.exe are the windowed variants;
rem  py.exe / python.exe would flash (or keep) a black console window.
rem
rem  IMPORTANT: this file is deliberately plain ASCII, and must stay that way.
rem  cmd.exe parses batch files by byte offset and loses its position when the
rem  file contains multi-byte characters, so it ends up executing fragments of
rem  comment lines as if they were commands.  On a GBK console that produced:
rem      'm' is not recognized as an internal or external command
rem  Keep all non-English explanation in README.md, never in here.
rem ===========================================================================

setlocal
set "HERE=%~dp0"

rem --- 1) pyw.exe: the windowed "py" launcher, no console window -------------
where pyw >nul 2>nul
if not errorlevel 1 (
    pyw -3 -c "import tkinter" >nul 2>nul
    if not errorlevel 1 (
        start "" pyw -3 "%HERE%llamagui.pyw"
        exit /b 0
    )
)

rem --- 2) pythonw.exe from PATH, no console window ---------------------------
where pythonw >nul 2>nul
if not errorlevel 1 (
    pythonw -c "import tkinter" >nul 2>nul
    if not errorlevel 1 (
        start "" pythonw "%HERE%llamagui.pyw"
        exit /b 0
    )
)

rem --- 3) py.exe: may briefly show a console window --------------------------
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import tkinter" >nul 2>nul
    if not errorlevel 1 (
        start "" py -3 "%HERE%llamagui.pyw"
        exit /b 0
    )
)

rem --- 4) python.exe: keeps the console open so errors stay readable ---------
where python >nul 2>nul
if not errorlevel 1 (
    python "%HERE%llamagui.pyw"
    exit /b %ERRORLEVEL%
)

echo.
echo No usable Python was found.
echo Install Python 3.9 or newer, including the "tcl/tk" option, then run:
echo     pythonw "%~dp0llamagui.pyw"
echo.
echo (If Python is already installed, make sure pythonw.exe is on PATH.)
echo.
pause
exit /b 1
