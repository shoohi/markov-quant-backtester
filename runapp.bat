@echo off
title Quant Backtester Launcher
echo =========================================
echo Starting Markov Chain Quant Backtester...
echo =========================================

cd C:\AfekaPython\quant_backtest
py -m streamlit run app.py

pause