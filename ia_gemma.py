import requests
import base64
import os
import json # Para lidar com a resposta JSON

# --- Configurações do Ollama ---
# O endereço padrão da API do Ollama. Certifique-se de que o Ollama está rodando.
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# --- MODELO ATUALIZADO PARA QWEN3 ---
OLLAMA_MODEL = "gemma3:4b" # O nome do modelo que você baixou no Ollama: gemma3:latest, qwen3:latest, gemma3n:latest

def consultar_ollama_local(instrucao: str, contexto: str, pergunta: str, imagem_path: str = None, modelo_ia: str = OLLAMA_MODEL) -> str:
    """
    Consulta a IA local (via Ollama) com uma instrução, contexto, pergunta e, opcionalmente, uma imagem.
    Os parâmetros textuais são concatenados em um prompt único.

    Args:
        instrucao (str): A instrução que a IA deve seguir (ex: "haja como um analista de RH").
        contexto (str): O texto base de informações que a IA irá usar para responder.
        pergunta (str): A pergunta específica que a IA irá responder.
        imagem_path (str, optional): O caminho para um arquivo de imagem local.
                                      Se fornecido, a imagem será enviada com a pergunta.
                                      Defaults to None.

    Returns:
        str: A resposta gerada pela IA. Retorna uma mensagem de erro em caso de falha.
    """
    
    if not instrucao or instrucao.strip() == "":
        instrucao = "Haja como um especialista no assunto perguntado.  Responda à pergunta a seguir usando **exclusivamente** as informações fornecidas no CONTEXTO"
    
    # --- CONSTRUÇÃO DO PROMPT FINAL ---
    # Concatenamos os três parâmetros de texto para formar o prompt completo para a IA.
    # Adicionei quebras de linha para melhor separação e clareza para o modelo.
    final_prompt = (
        f"Instrução: {instrucao}\n\n"
        f"Contexto: {contexto}\n\n"
        f"Pergunta: {pergunta}"
    )

    if modelo_ia is None or modelo_ia.strip() == "":
        modelo_ia = OLLAMA_MODEL
    
    print(f"DEBUG: Iniciando consulta ao modelo {modelo_ia}.")
    print(f"DEBUG: Prompt final enviado (início): '{final_prompt[:200]}...'") # Log do início do prompt
    
    payload = {
        "model": modelo_ia,
        "prompt": final_prompt, # Agora usamos o prompt final concatenado
        "stream": False # Define como False para obter a resposta completa de uma vez
    }

    if imagem_path:
        if not os.path.exists(imagem_path):
            return f"Erro: Imagem não encontrada no caminho especificado: {imagem_path}"
        
        try:
            with open(imagem_path, "rb") as image_file:
                # Codifica a imagem em Base64
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Adiciona a imagem codificada ao payload
            payload["images"] = [encoded_image]
            print(f"DEBUG: Imagem {os.path.basename(imagem_path)} codificada e adicionada ao payload.")
        except Exception as e:
            return f"Erro ao processar a imagem: {e}"

    headers = {'Content-Type': 'application/json'}

    try:
        print(f"DEBUG: Enviando requisição para {OLLAMA_API_URL} com o modelo {modelo_ia}...")
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Lança uma exceção para códigos de status de erro (4xx ou 5xx)
        
        response_data = response.json()
        
        if "response" in response_data:
            return response_data["response"]
        elif "error" in response_data:
            return f"Erro da API Ollama: {response_data['error']}"
        else:
            return f"Resposta inesperada da API Ollama: {response_data}"

    except requests.exceptions.ConnectionError:
        return "Erro de conexão: Certifique-se de que o Ollama está rodando e acessível em http://localhost:11434."
    except requests.exceptions.HTTPError as http_err:
        return f"Erro HTTP: {http_err} - Resposta: {response.text}"
    except json.JSONDecodeError:
        return f"Erro ao decodificar a resposta JSON da API Ollama. Resposta: {response.text}"
    except Exception as e:
        return f"Ocorreu um erro inesperado: {e}"

# --- Bloco de Teste Interativo ---
if __name__ == "__main__":
    print(f"--- Teste Interativo da IA via Ollama ({OLLAMA_MODEL}) ---")
    print("Certifique-se de que o Ollama está rodando e o modelo '{OLLAMA_MODEL}' foi baixado.")

    while True:
        print("\n--- Entrada de Dados para a IA ---")
        instrucao_usuario = input("Digite a INSTRUÇÃO para a IA (ex: 'haja como um analista de RH e responda apenas com base no contexto'): \n> ").strip()
        contexto_usuario = input("Digite o CONTEXTO de informações para a IA (deixe em branco se não houver): \n> ").strip()
        pergunta_usuario = input("Digite a PERGUNTA para a IA (ou 'sair' para encerrar): \n> ").strip()
        modelo_ia = input(f"Digite o NOME DO MODELO (padrão: '{OLLAMA_MODEL}'): \n> ").strip()

        if pergunta_usuario.lower() == 'sair':
            print("Encerrando o teste interativo.")
            break

        # Define um contexto padrão se o usuário não fornecer um
        if not contexto_usuario:
            contexto_usuario = "Nenhum contexto adicional foi fornecido."

        caminho_imagem = input("Digite o caminho completo para uma imagem (opcional, deixe em branco se não houver): \n> ").strip()

        if caminho_imagem == "":
            caminho_imagem = None
        elif not os.path.exists(caminho_imagem):
            print(f"ATENÇÃO: O caminho da imagem '{caminho_imagem}' não foi encontrado. A consulta será feita sem a imagem.")
            caminho_imagem = None # Garante que a IA não tente usar um caminho inválido

        print("\nProcessando sua solicitação...\n")
        # --- CHAMADA DA FUNÇÃO COM NOVOS PARÂMETROS ---
        resposta_ia = consultar_ollama_local(instrucao_usuario, contexto_usuario, pergunta_usuario, caminho_imagem, modelo_ia)
        
        print("\n--- Resposta da IA ---")
        print(resposta_ia)
        print("---------------------\n")

        while True:
            continuar = input("Deseja continuar? (S/N): ").strip().lower()
            if continuar in ['s', 'n']:
                break
            else:
                print("Opção inválida. Por favor, digite 'S' para Sim ou 'N' para Não.")
        
        if continuar == 'n':
            print("Teste interativo encerrado.")
            break