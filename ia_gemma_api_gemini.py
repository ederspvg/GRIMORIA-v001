# ia_gemma_api_gemini.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image # <--- ADICIONADO: Importação para lidar com imagens

# --- Configuração da API Google Generative AI ---
# Carrega as variáveis de ambiente do arquivo .env
# Certifique-se de ter um arquivo 'ambiente.env' no mesmo diretório
# contendo GEMINI_API_KEY="SUA_CHAVE_AQUI"
load_dotenv(dotenv_path='ambiente.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("A variável de ambiente 'GEMINI_API_KEY' não está configurada no arquivo 'ambiente.env'.")

genai.configure(api_key=GEMINI_API_KEY)

# --- Configurações do Modelo ---
# Modelo Gemma 3 de 27 bilhões de parâmetros, ajustado para instruções.
# Este é o modelo padrão textual. Ele será sobreescrito se uma imagem for fornecida na consulta.
API_MODEL = "gemma-3-27b-it"  #gemma-3-27b-it #gemma-3-4b-it #gemma-3-12b-it #gemma-3n-e4b-it #gemma3:27b-it-fp16 (55GB)

def consultar_gemma_api_gemini(instrucao: str, contexto: str, pergunta: str, imagem_path: str = None, modelo_ia: str = API_MODEL) -> str:
    """
    Consulta o modelo Gemma/Gemini via API do Google com uma instrução, contexto, pergunta
    e, opcionalmente, uma imagem. Permite especificar o modelo de IA.
    Se uma imagem for fornecida, um modelo multimodal é garantido.

    Args:
        instrucao (str): A instrução que a IA deve seguir (ex: "haja como um analista de RH").
        contexto (str): O texto base de informações que a IA irá usar para responder.
        pergunta (str): A pergunta específica que a IA irá responder.
        imagem_path (str, optional): O caminho para um arquivo de imagem local (JPG, PNG, WEBP, etc.).
                                      Se fornecido, a imagem será enviada com a pergunta e
                                      um modelo multimodal será usado. Defaults to None.
        modelo_ia (str, optional): O nome específico do modelo de IA a ser usado (ex: "gemini-1.5-flash-latest").
                                   Se não for fornecido, usa API_MODEL como padrão.
                                   Se uma imagem_path for fornecida, o modelo será forçado
                                   a um multimodal se o modelo_ia padrão for textual.

    Returns:
        str: A resposta gerada pela IA. Retorna uma mensagem de erro em caso de falha.
    """
    prompt_parts = []
    if instrucao:
        prompt_parts.append(instrucao)
    if contexto:
        prompt_parts.append(contexto)
    if pergunta:
        prompt_parts.append(pergunta)
    
    text_prompt = "\n".join(prompt_parts)

    contents = []
    model_to_use = modelo_ia # Inicia com o modelo_ia passado ou o padrão API_MODEL

    if imagem_path:
        try:
            # Tenta carregar a imagem
            image = PIL.Image.open(imagem_path)
            contents.append(image)
            
            # Se imagem_path é fornecido, garantimos que o modelo seja multimodal.
            # Se o modelo_ia passado não começar com "gemini-1.5-" (indicando um modelo multimodal),
            # ou se for o API_MODEL textual padrão, forçamos para 'gemini-1.5-flash-latest'.
            # if not model_to_use.startswith("gemini-1.5-"): # Simplificado para pegar modelos 1.5 flash/pro
                #  print(f"[!] Aviso: Imagem fornecida. Forçando modelo para 'gemini-1.5-flash-latest' (multimodal) ao invés de '{model_to_use}'.")
                #  model_to_use = API_MODEL # Força um modelo multimodal
            # Se o modelo_ia já é um gemini-1.5-*, assumimos que é multimodal e o mantemos.
        except FileNotFoundError:
            return f"Erro: Imagem não encontrada no caminho especificado: {imagem_path}"
        except Exception as e:
            return f"Erro ao carregar imagem: {e}"
    
    contents.append(text_prompt)

    try:
        model = genai.GenerativeModel(model_to_use)
        print(f"  [i] Chamando API Gemini (modelo: {model_to_use})...")
        response = model.generate_content(contents)
        print(f"  [+] Resposta do Gemini recebida.")
        
        # A API pode retornar em 'text' ou 'parts' dependendo do tipo de resposta.
        # Tentamos primeiro 'text', depois 'parts' se for o caso de respostas complexas (multimodais).
        if hasattr(response, 'text'):
            return response.text
        # Em casos de respostas multimodais com várias partes, pode ser necessário iterar sobre response.parts
        # No entanto, para a maioria das descrições simples, 'text' é suficiente.
        # Se 'text' não estiver disponível e houver 'parts', tentamos extrair o texto de lá.
        elif hasattr(response, 'parts') and response.parts:
            extracted_text = ""
            for part in response.parts:
                if hasattr(part, 'text'):
                    extracted_text += part.text
            if extracted_text:
                return extracted_text
            else:
                return "Resposta da IA vazia ou em formato de partes sem texto."
        else:
            return "Resposta da IA vazia ou em formato inesperado."

    except Exception as e:
        print(f"  [-] Erro ao chamar a API Gemini: {e}")
        error_message = str(e)
        if hasattr(e, 'message'): error_message = e.message
        if "block_reason: SAFETY" in error_message or "response was blocked" in error_message:
            return "A resposta foi bloqueada devido às configurações de segurança."
        elif "token" in error_message.lower():
            return f"Ocorreu um erro relacionado ao limite de tokens: {error_message}"
        else:
            return f"Ocorreu um erro ao gerar a resposta pela IA: {error_message}"
        
#------------------------------------------------------------------------------------------------------
# Bloco de Teste interativo
# 
# --- Bloco de Teste Interativo ---
if __name__ == "__main__":
    print(f"--- Teste Interativo de IA via API Google Generative AI (Modelo: {API_MODEL}) ---")
    print("Certifique-se de que sua chave de API GEMINI_API_KEY está configurada no arquivo 'ambiente.env'.")
    print("\nCertifique-se de escolher um modelo multimodal para garantir que será possível analisar imagens quando necessário.")

    while True:
        print("\n--- Entrada de Dados para a IA ---")
        instrucao_usuario = input("Digite a INSTRUÇÃO para a IA (ex: 'Haja como um especialista em IA'): \n> ").strip()
        contexto_usuario = input("Digite o CONTEXTO de informações para a IA (deixe em branco se não houver): \n> ").strip()
        pergunta_usuario = input("Digite a PERGUNTA para a IA (ou 'sair' para encerrar): \n> ").strip()
        caminho_imagem_teste = input("Digite o caminho completo para uma imagem de teste (opcional, deixe em branco para pular): ").strip()

        if pergunta_usuario.lower() == 'sair':
            print("Encerrando o teste interativo.")
            break

        # Se o contexto estiver vazio, podemos definir um valor padrão para evitar prompt em branco
        if not contexto_usuario:
            contexto_usuario = "Nenhum contexto adicional fornecido."

        print("\nProcessando sua solicitação...\n")
        # Chamada da função com os três novos parâmetros
        resposta_ia = consultar_gemma_api_gemini(instrucao_usuario, contexto_usuario, pergunta_usuario,caminho_imagem_teste)
        
        print("\n--- Resposta da IA ---")
        print(f"Descrição da imagem: {resposta_ia}")
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