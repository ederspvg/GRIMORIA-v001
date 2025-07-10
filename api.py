# api.txt (versão final atualizada)

from flask import Flask, request, jsonify
import rag
import ia_gemma_api_gemini as ia_gemini
import ia_gemma as ia_local 

app = Flask(__name__)

# Instancia o sistema RAG uma única vez na inicialização
# O __init__ da classe RAG já cuida de iniciar o servidor do banco de dados
sistema_rag = rag.SistemaRAG()

@app.route('/upload_e_criar_colecao', methods=['POST'])
def upload_e_criar_colecao():
    """Recebe o upload de um ou mais arquivos e um tema."""
    tema = request.form.get('tema')
    if not tema:
        return jsonify({"erro": "O parâmetro 'tema' é obrigatório no formulário."}), 400

    if 'files' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"erro": "Nenhum arquivo selecionado."}), 400

    try:
        resultado = sistema_rag.processar_uploads_e_criar_colecoes(files, tema)
        if resultado.get("sucesso"):
            return jsonify({"mensagem": resultado.get("mensagem")}), 200
        else:
            return jsonify({"erro": resultado.get("mensagem")}), 500
    except Exception as e:
        return jsonify({"erro": f"Um erro inesperado ocorreu na API: {str(e)}"}), 500


@app.route('/criar_colecao', methods=['POST'])
def criar_colecao():
    """Cria coleções a partir de um diretório com arquivos."""
    diretorio = request.json.get('diretorio')
    if not diretorio:
        return jsonify({"erro": "O parâmetro 'diretorio' é obrigatório."}), 400
    try:
        sistema_rag.criar_colecoes_da_pasta(diretorio)
        return jsonify({"mensagem": f"Coleções criadas a partir do diretório '{diretorio}'."}), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/listar_colecoes', methods=['GET'])
def listar_colecoes():
    """Lista todas as coleções existentes no sistema, já ordenadas."""
    try:
        lista_de_colecoes = sistema_rag.listar_colecoes()
        return jsonify(lista_de_colecoes), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/deletar_colecao', methods=['DELETE'])
def deletar_colecao():
    """Deleta uma coleção específica pelo ID."""
    id_colecao = request.json.get('id_colecao')
    if not id_colecao:
        return jsonify({"erro": "O parâmetro 'id_colecao' é obrigatório."}), 400
    try:
        resultado = sistema_rag.deletar_colecao_por_nome(id_colecao)
        if resultado:
            return jsonify({"mensagem": f"Coleção '{id_colecao}' deletada com sucesso."}), 200
        else:
            return jsonify({"erro": f"Coleção '{id_colecao}' não encontrada."}), 404
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/zerar_todas_colecoes', methods=['DELETE'])
def zerar_todas_colecoes():
    """Remove todas as coleções do sistema."""
    try:
        sistema_rag.zerar_todas_colecoes()
        return jsonify({"mensagem": "Todas as coleções foram removidas com sucesso."}), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# --- ENDPOINT ATUALIZADO ---
@app.route('/consultar', methods=['POST'])
def consultar():
    """
    Consulta o sistema RAG com uma pergunta e parâmetros opcionais.
    """
    # 1. Obter o parâmetro obrigatório
    pergunta = request.json.get('pergunta')
    if not pergunta:
        return jsonify({"erro": "O parâmetro 'pergunta' é obrigatório."}), 400

    # 2. Obter todos os parâmetros opcionais, com valores padrão
    #    Os valores padrão são os mesmos da assinatura do método em rag_4.txt,
    #    garantindo consistência se o cliente não enviar o campo.
    usar_ia_local = request.json.get('usar_ia_local', False)
    instrucao = request.json.get('instrucao', "")
    modelo_de_pensamento = request.json.get('modelo_de_pensamento', "gemma-3-27b-it")
    n_results_per_colecao = request.json.get('n_results_per_colecao', 10)
    max_distance_threshold = request.json.get('max_distance_threshold', 0.8)
    
    # Parâmetros de caminho de arquivo (tratados como strings)
    pdf_path = request.json.get('pdf_path', None)
    imagem_path = request.json.get('imagem_path', None)

    try:
        # 3. Chamar o método RAG com todos os parâmetros
        resposta = sistema_rag.consultar_multiplas_colecoes(
            pergunta=pergunta,
            usar_ia_local=usar_ia_local,
            instrucao=instrucao,
            pdf_path=pdf_path,
            imagem_path=imagem_path,
            modelo_de_pensamento=modelo_de_pensamento,
            n_results_per_colecao=n_results_per_colecao,
            max_distance_threshold=max_distance_threshold
        )
        return jsonify({"resposta": resposta}), 200
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado durante a consulta: {str(e)}"}), 500
    
# --- Endpoint de Chamada Direta à IA ---

@app.route('/chamar_ia_direto', methods=['POST'])
def chamar_ia_direto():
    """Endpoint para chamar a IA generativa diretamente, com opção de local ou API."""
    data = request.get_json()
    if not data: return jsonify({"erro": "Corpo da requisição JSON está vazio."}), 400
        
    pergunta = data.get('pergunta')
    if not pergunta: return jsonify({"erro": "O parâmetro 'pergunta' é obrigatório."}), 400
        
    instrucao = data.get('instrucao', "")
    contexto = data.get('contexto', "")
    imagem_path = data.get('imagem_path', None)
    usar_ia_local = data.get('usar_ia_local', False)
    modelo_ia = data.get('modelo_ia') # Pega o modelo enviado pelo cliente

    try:
        if usar_ia_local:
            print(f"[API] Chamada direta para IA LOCAL (modelo: {modelo_ia}). Pergunta: '{pergunta[:50]}...'")
            # Define um modelo padrão para Ollama se nenhum for enviado
            if not modelo_ia: modelo_ia = "gemma3:latest"
            resposta_ia = ia_local.consultar_ollama_local(
                instrucao=instrucao, contexto=contexto, pergunta=pergunta,
                imagem_path=imagem_path, modelo_ia=modelo_ia
            )
        else:
            print(f"[API] Chamada direta para IA API (modelo: {modelo_ia}). Pergunta: '{pergunta[:50]}...'")
            # Define um modelo padrão para a API se nenhum for enviado
            if not modelo_ia: modelo_ia = ia_gemini.API_MODEL
            resposta_ia = ia_gemini.consultar_gemma_api_gemini(
                instrucao=instrucao, contexto=contexto, pergunta=pergunta,
                imagem_path=imagem_path, modelo_ia=modelo_ia
            )
            
        return jsonify({"resposta": resposta_ia}), 200
    except Exception as e:
        return jsonify({"erro": f"Erro ao chamar a IA: {str(e)}"}), 500

# RETORNA UMA LISTA DE MODELOS DISPONIVEIS NO GEMINI
# Este endpoint permite que a interface Streamlit ou qualquer cliente consulte os modelos disponíveis

@app.route('/listar_modelos_ia', methods=['GET'])
def listar_modelos_ia():
    """
    Endpoint para listar os modelos de IA disponíveis na API do Google.
    """
    try:
        print("[API] Solicitada a lista de modelos de IA disponíveis.")
        # Chama a função diretamente do módulo importado
        modelos_disponiveis_str = ia_gemini.consultar_modelos_gemini_disponiveis()
        
        # A função original retorna uma string formatada. Vamos dividi-la em uma lista
        # para um JSON mais estruturado e útil para as aplicações clientes.
        lista_modelos = []
        if "Modelos disponíveis:" in modelos_disponiveis_str:
            # Pega as linhas, remove o "- " inicial e linhas vazias
            lista_modelos = [
                line.strip().replace("- ", "") 
                for line in modelos_disponiveis_str.split('\n') 
                if line.strip() and line.strip().startswith("-")
            ]
        
        return jsonify({"modelos": lista_modelos}), 200

    except Exception as e:
        return jsonify({"erro": f"Um erro inesperado ocorreu ao listar os modelos: {str(e)}"}), 500


if __name__ == '__main__':
    # Usar host='0.0.0.0' torna a API acessível na sua rede local,
    # o que é útil para a interface Streamlit rodar em outra máquina.
    app.run(host='0.0.0.0', debug=True)
    