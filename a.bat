@echo off

:: Create main project folder
mkdir CryptoTrendPredictor
cd CryptoTrendPredictor

:: Create subdirectories
mkdir data\historical data\real_time model\notebooks model\scripts model\models backend\app backend\tests frontend\public frontend\src frontend\tests docs

:: Create .md files
echo # Setup Instructions > docs\setup.md
echo # Usage Instructions > docs\usage.md

:: Create other root files
echo # CryptoTrendPredictor > README.md
type nul > .gitignore
type nul > requirements.txt
echo from setuptools import setup, find_packages ^> setup.py
echo setup(^>^> setup.py
echo     name="CryptoTrendPredictor", ^>^> setup.py
echo     version="0.1", ^>^> setup.py
echo     packages=find_packages(), ^>^> setup.py
echo ) ^>^> setup.py

:: Output
echo.
echo Directory structure created under: 
cd
pause