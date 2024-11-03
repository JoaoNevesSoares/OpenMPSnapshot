#!/bin/bash

# Defina o diretório alvo
TARGET_DIR="repCPP/"

# Use o comando find para localizar e deletar arquivos que não correspondem às extensões especificadas
find "$TARGET_DIR" -type f ! \( \
    -iname "*.c" -o \
    -iname "*.h" -o \
    -iname "*.cpp" -o \
    -iname "*.hpp" -o \
    -iname "*.cxx" -o \
    -iname "*.hxx" -o \
    -iname "*.inl" \
\) -exec rm -f {} +

echo "Arquivos não correspondentes foram removidos."
