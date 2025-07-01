from flask import Flask, request, jsonify
import rag  # Importa o script rag.py

app = Flask(__name__)

# Instancia o sistema RAG
sistema_rag = rag.SistemaRAG()

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
    """Lista todas as coleções existentes no sistema."""
    try:
        colecoes = []
        for nome_colecao in sistema_rag.lista_nomes_colecoes:
            colecao = sistema_rag.client.get_collection(name=nome_colecao, embedding_function=rag.embedding_function)
            metadados = colecao.get(include=['metadatas'])['metadatas']
            if not metadados:
                continue

            for meta in metadados:
                colecoes.append({
                    "id": nome_colecao,
                    "nome_arquivo": meta.get("nome_arquivo_original", "N/A"),
                    "tema": meta.get("tema", "N/A")
                })
        return jsonify(colecoes), 200
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

@app.route('/consultar', methods=['POST'])
def consultar():
    """Consulta o sistema RAG com base em uma pergunta."""
    pergunta = request.json.get('pergunta')
    usar_ia_local = request.json.get('usar_ia_local', False)
    n_results_per_colecao = request.json.get('n_results_per_colecao', 10)
    max_distance_threshold = request.json.get('max_distance_threshold', 0.8)

    if not pergunta:
        return jsonify({"erro": "O parâmetro 'pergunta' é obrigatório."}), 400

    try:
        resposta = sistema_rag.consultar_multiplas_colecoes(
            pergunta=pergunta,
            usar_ia_local=usar_ia_local,
            n_results_per_colecao=n_results_per_colecao,
            max_distance_threshold=max_distance_threshold
        )
        return jsonify({"resposta": resposta}), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)