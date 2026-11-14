@echo off
rem Drop RGBI scans or folders on this file: exports 2400 px wide negative + infrared training
rem pairs into a "training" folder next to them (see extract-ir.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0extract-ir.ps1" %*
pause
