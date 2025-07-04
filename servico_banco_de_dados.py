# servico_banco_de_dados.py (versão final com lançador .bat)

import subprocess
import time
import requests
import os
import platform
from typing import Tuple
import parametros_globais as OneRing

# --- Configurações ---
HOST = "localhost"
PORT = 8000
PERSIST_DIRECTORY = OneRing.PASTA_BANCO
LOG_FILE = "chroma_server.log"
BATCH_FILE_NAME = "start_chroma_server.bat"

def verificar_servico_chroma(host: str = HOST, port: int = PORT) -> bool:
    """Verifica se o servidor ChromaDB está respondendo na porta especificada."""
    try:
        requests.get(f"http://{host}:{port}/api/v1/heartbeat", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False

def iniciar_servico_chroma(path: str = PERSIST_DIRECTORY, host: str = HOST, port: int = PORT, log_file: str = LOG_FILE) -> bool:
    """Inicia o servidor ChromaDB usando um script .bat para desacoplamento total no Windows."""
    print(f"[i] Tentando iniciar o servidor ChromaDB via lançador '{BATCH_FILE_NAME}'...")

    # Garante que o caminho de persistência seja absoluto para o .bat
    absolute_persist_path = os.path.abspath(path)
    print(f"[i] Banco de dados persistido em: '{absolute_persist_path}'")
    
    # Cria o conteúdo do arquivo .bat
    # @echo off: não mostra os comandos no console
    # start /B: inicia o comando em segundo plano, sem uma nova janela de console
    # >>: redireciona a saída (stdout e stderr) para o arquivo de log
    batch_content = f"""
@echo off
echo Iniciando ChromaDB Server...
start /B chroma run --host {host} --port {port} --path "{absolute_persist_path}" >> {log_file} 2>&1
"""
    # Salva o arquivo .bat no mesmo diretório do script
    try:
        with open(BATCH_FILE_NAME, "w") as f:
            f.write(batch_content)
    except Exception as e:
        print(f"[-] Erro crítico ao criar o script lançador '{BATCH_FILE_NAME}': {e}")
        return False

    # Executa o arquivo .bat
    try:
        # Executa o .bat e não espera por ele. O .bat irá lançar o Chroma e encerrar.
        subprocess.Popen([BATCH_FILE_NAME], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"[-] Erro ao executar o script lançador '{BATCH_FILE_NAME}': {e}")
        return False

    # Loop de verificação robusto
    max_tentativas = 15
    tempo_espera = 2
    print("[i] Aguardando o servidor ficar pronto...")
    
    for tentativa in range(max_tentativas):
        print(f"  ... verificação (tentativa {tentativa + 1}/{max_tentativas})")
        if verificar_servico_chroma(host, port):
            print("[+] Servidor ChromaDB está online e pronto!")
            return True
        time.sleep(tempo_espera)
    
    print(f"[-] Falha ao iniciar o servidor após {max_tentativas * tempo_espera} segundos.")
    print(f"[-] Verifique o arquivo de log '{log_file}' e se o Firewall do Windows não está bloqueando o processo 'chroma'.")
    return False

def garantir_servico_ativo():
    """Função principal que orquestra a verificação e o início do serviço."""
    if verificar_servico_chroma():
         print("[+] Servidor ChromaDB já está ativo.")
    else:
        iniciar_servico_chroma()

# Bloco para teste direto
if __name__ == '__main__':
    print("--- Verificador e Iniciador de Serviço ChromaDB ---")
    garantir_servico_ativo()