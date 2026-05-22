@echo off
echo Downloading local models for Corporate Bot...
echo.

echo Downloading language model qwen3.5:9b...
ollama pull qwen3.5:9b

echo Downloading embedding model nomic-embed-text...
ollama pull nomic-embed-text

echo.
echo Download complete!
pause
