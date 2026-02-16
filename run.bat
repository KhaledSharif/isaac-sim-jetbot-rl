@echo off
setlocal enabledelayedexpansion
REM Universal Isaac Sim Python script launcher
REM
REM Usage:
REM   run.bat                              Run jetbot_keyboard_control.py
REM   run.bat --enable-recording           Run with recording enabled
REM   run.bat replay.py demos\file.npz     Run replay script
REM   run.bat train_bc.py demos\file.npz   Run training script

set "ISAAC_PYTHON=C:\isaac-sim\python.bat"

REM Check if Isaac Sim Python exists
if not exist "%ISAAC_PYTHON%" (
    echo Error: Isaac Sim Python not found at %ISAAC_PYTHON%
    echo Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation.
    exit /b 1
)

REM Determine which script to run
set "ARG1=%~1"
if "%ARG1%"=="" goto run_default

REM Check if first arg ends in .py
echo %ARG1%| findstr /i "\.py$" >nul 2>&1
if errorlevel 1 goto run_default


REM First arg is a .py file - run that script with remaining args
set "SCRIPT=%ARG1%"

REM If script doesn't exist as-is, try src\ directory
if not exist "%SCRIPT%" set "SCRIPT=src\%SCRIPT%"

REM Build remaining args (skip first arg since shift doesn't affect %*)
set "REMAINING="
shift
:argloop
if "%~1"=="" goto argdone
set "REMAINING=!REMAINING! %1"
shift
goto argloop
:argdone

echo Running %SCRIPT% with Isaac Sim's Python...
call "%ISAAC_PYTHON%" "%SCRIPT%" %REMAINING%
goto :eof

:run_default
REM No .py file specified - run main app with all args
echo Running Jetbot keyboard control with Isaac Sim's Python...
call "%ISAAC_PYTHON%" src\jetbot_keyboard_control.py %*
