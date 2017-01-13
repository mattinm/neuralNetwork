@echo off

setlocal

set cwd=%~dp0

where /q perl
IF ERRORLEVEL 1 (
    ECHO Perl cannot be found. Please ensure it is installed and placed in your PATH.
	PAUSE
    EXIT /B
) 

where /q nasm
IF ERRORLEVEL 1 (
    ECHO nasm cannot be found. Please ensure it is installed and placed in your PATH.
	PAUSE
    EXIT /B
) 


set OPENSSL_VERSION=1.1.1-dev
set SEVENZIP="C:\Program Files\7-Zip\7z.exe"
set VS2015="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\vcvars32.bat"
set VS2015_AMD64="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"

set OPENSSL_DIR="%cwd%..\..\externals\openssl"

IF EXIST %OPENSSL_DIR% GOTO OPENSSL_SOURCE
ECHO pull in the openssl submodule
PAUSE
EXIT \B
:OPENSSL_SOURCE


set OPENSSL32_DIR="%cwd%openssl_win32"

REM Remove openssl source directories
IF NOT EXIST %OPENSSL32_DIR% GOTO NO_WIN32_SOURCE
DEL %OPENSSL32_DIR% /Q /F /S
RMDIR /S /Q %OPENSSL32_DIR%
:NO_WIN32_SOURCE
MKDIR %OPENSSL32_DIR%


set OPENSSL64_DIR="%cwd%openssl_win64"

IF NOT EXIST %OPENSSL64_DIR% GOTO NO_WIN64_SOURCE
DEL %OPENSSL64_DIR% /Q /F /S
RMDIR /S /Q %OPENSSL64_DIR%
:NO_WIN64_SOURCE
MKDIR %OPENSSL64_DIR%


# copy the source
XCOPY /E /H /Y /K %OPENSSL_DIR% %OPENSSL32_DIR%
XCOPY /E /H /Y /K %OPENSSL_DIR% %OPENSSL64_DIR%

cd %OPENSSL32_DIR%
CALL %VS2015%
cd %OPENSSL32_DIR%

ECHO "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-DLL-VS2015"
perl Configure VC-WIN32 --prefix="%cwd%openssl-%OPENSSL_VERSION%-32bit-release-DLL-VS2015"
nmake
nmake test
nmake install

perl Configure debug-VC-WIN32 --prefix="%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-DLL-VS2015"
nmake
nmake test
nmake install

perl Configure VC-WIN32 --prefix="%cwd%openssl-%OPENSSL_VERSION%-32bit-release-static-VS2015"
nmake
nmake test
nmake install

perl Configure debug-VC-WIN32 --prefix="%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-static-VS2015"
nmake
nmake test
nmake install

cd %OPENSSL64_DIR%
CALL %VS2015_AMD64%
cd %OPENSSL64_DIR%

perl Configure VC-WIN64A --prefix="%cwd%openssl-%OPENSSL_VERSION%-64bit-release-DLL-VS2015"
nmake
nmake test
nmake install

perl Configure debug-VC-WIN64A --prefix="%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-DLL-VS2015"
nmake
nmake test
nmake install

cd \openssl-src-win64-VS2015
perl Configure VC-WIN64A --prefix="%cwd%openssl-%OPENSSL_VERSION%-64bit-release-static-VS2015"
nmake
nmake test
nmake install

perl Configure debug-VC-WIN64A --prefix="%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-static-VS2015"
nmake
nmake test
nmake install

cd \
python copy_openssl_pys.py

%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-DLL-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-DLL-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-DLL-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-DLL-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-DLL-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-DLL-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-DLL-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-DLL-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-static-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-static-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-static-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-static-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-static-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-static-VS2015\*"
%SEVENZIP% a -r "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-static-VS2015.7z" "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-static-VS2015\*"

DEL "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-DLL-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-DLL-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-DLL-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-DLL-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-static-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-static-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-static-VS2015" /Q /F /S
DEL "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-static-VS2015" /Q /F /S

RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-DLL-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-DLL-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-DLL-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-DLL-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-32bit-debug-static-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-32bit-release-static-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-64bit-debug-static-VS2015"
RMDIR /S /Q "%cwd%openssl-%OPENSSL_VERSION%-64bit-release-static-VS2015"

PAUSE