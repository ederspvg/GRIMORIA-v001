# rag_3.txt (versão atualizada e corrigida)

import parametros_globais as OneRing
import utilitarios as Canivete
import prompts_ia as Persona
import uuid  # Para gerar IDs únicos para os chunks
import ia_gemma as Gemma_IA
import ia_gemma_api_gemini as Gemma_IA_API
import os
import json  # Para persistir a lista de coleções
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PIL import Image
from typing import List, Tuple
import shutil
import pdfplumber
import pandas as pd
import csv
import servico_banco_de_dados as db_service  # <-- 1. Importar o novo script

# Configuração da API Gemini
load_dotenv(dotenv_path='ambiente.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# --- Configurações do ChromaDB ---
# O caminho de persistência é usado tanto pelo servidor quanto para o arquivo de metadados JSON.
PERSIST_DIRECTORY = OneRing.PASTA_BANCO
PERSIST_PASTA_BIBLIOTECA = OneRing.PASTA_BIBLIOTECA
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Funções de embedding
class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

# ---------------------------------------------------------------------------------------------------------------------------
# Sistema RAG (Retrieval Augmented Generation) - Classe Principal
# ---------------------------------------------------------------------------------------------------------------------------
class SistemaRAG:
    def __init__(self, 
                 persist_directory: str = PERSIST_DIRECTORY, 
                 host: str = CHROMA_HOST, 
                 port: int = CHROMA_PORT):
        
        # 2. Garantir que o servidor do banco de dados está ativo ANTES de tentar conectar
        print("\n[DB] Verificando status do servidor ChromaDB...")
        db_service.garantir_servico_ativo()
        
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        
        # 3. Mudar para HttpClient para se conectar ao servidor de forma segura
        print(f"[DB] Conectando ao servidor ChromaDB em http://{self.host}:{self.port}...")
        self.client = chromadb.HttpClient(host=self.host, port=self.port)

        # A lógica de gerenciamento de metadados (lista de coleções, temas) permanece.
        self.lista_colecoes_file = os.path.join(persist_directory, "lista_colecoes.json")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.lista_nomes_colecoes = self._carregar_lista_colecoes()
        self.temas_disponiveis = self._carregar_temas_disponiveis()
        
        print(f"[i] SistemaRAG inicializado e conectado ao servidor ChromaDB.")
        print(f"[i] {len(self.lista_nomes_colecoes)} coleções de documentos registradas.")

    # O RESTANTE DA CLASSE NÃO PRECISA DE ALTERAÇÕES
    # A interface do HttpClient é a mesma do PersistentClient,
    # então todos os seus métodos de manipulação de coleções
    # funcionarão como antes, mas agora de forma segura e sem corrupção de dados.

    def _carregar_lista_colecoes(self) -> List[str]:
        if os.path.exists(self.lista_colecoes_file):
            try:
                with open(self.lista_colecoes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[-] Erro: Arquivo '{self.lista_colecoes_file}' corrompido ou vazio. Iniciando com lista vazia.")
                return []
        return []

    def _salvar_lista_colecoes(self):
        with open(self.lista_colecoes_file, 'w', encoding='utf-8') as f:
            json.dump(self.lista_nomes_colecoes, f, indent=4)

    def _carregar_temas_disponiveis(self) -> List[str]:
        """Carrega a lista de temas disponíveis a partir das coleções existentes."""
        temas = set()
        for nome_colecao in self.lista_nomes_colecoes:
            try:
                colecao = self.client.get_collection(name=nome_colecao, embedding_function=embedding_function)
                for metadata in colecao.get(include=['metadatas'])['metadatas']:
                    tema = metadata.get("tema")
                    if tema:
                        temas.add(tema)
            except Exception as e:
                # Este erro é esperado se a coleção foi deletada manualmente no servidor
                # mas o JSON ainda não foi atualizado. Não é crítico.
                # print(f"[-] Aviso ao carregar temas da coleção '{nome_colecao}': {e}")
                pass
        return list(temas)

    def _obter_proximo_nome_colecao(self) -> str:
        """Gera um nome numérico sequencial para a próxima coleção."""
        if not self.lista_nomes_colecoes:
            return "0001"
        numeros_existentes = []
        for nome in self.lista_nomes_colecoes:
            try:
                numeros_existentes.append(int(nome))
            except ValueError:
                pass
        if numeros_existentes:
            ult_num = max(numeros_existentes)
            return str(ult_num + 1).zfill(4)
        else:
            return "0001"

    def _dividir_texto_em_chunks(self, texto: str, tamanho_chunk: int = 500, sobreposicao: int = 50) -> List[str]:
        if sobreposicao >= tamanho_chunk:
            sobreposicao = tamanho_chunk - 1
        chunks = []
        palavras = texto.split()
        num_palavras = len(palavras)
        i = 0
        while i < num_palavras:
            fim = min(i + tamanho_chunk, num_palavras)
            chunk = " ".join(palavras[i:fim])
            chunks.append(chunk)
            if fim == num_palavras:
                break
            i += (tamanho_chunk - sobreposicao)
            if i < 0:
                i = 0
            if i >= num_palavras and fim < num_palavras:
                if len(" ".join(palavras[fim:]).strip()) > 0:
                    chunks.append(" ".join(palavras[fim:]))
                break
        return chunks

    def _ler_pdf(self, caminho_arquivo: str) -> str:
        try:
            with pdfplumber.open(caminho_arquivo) as pdf:
                texto = "".join([page.extract_text() or "" for page in pdf.pages])
            return texto if texto.strip() else self._ler_pdf_pypdf2(caminho_arquivo)
        except Exception as e:
            print(f"[-] Erro ao ler PDF com pdfplumber: {e}. Tentando PyPDF2...")
            return self._ler_pdf_pypdf2(caminho_arquivo)

    def _ler_pdf_pypdf2(self, caminho_arquivo: str) -> str:
        try:
            reader = PdfReader(caminho_arquivo)
            texto = "".join([page.extract_text() or "" for page in reader.pages])
            return texto
        except Exception as e:
            print(f"[-] Erro ao ler PDF com PyPDF2: {e}")
            return ""

    def _ler_txt(self, caminho_arquivo: str) -> str:
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(caminho_arquivo, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"[-] Erro ao ler TXT: {e}")
            return ""

    def _ler_docx(self, caminho_arquivo: str) -> str:
        try:
            doc = Document(caminho_arquivo)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"[-] Erro ao ler DOCX: {e}")
            return ""

    def _ler_csv(self, caminho_arquivo: str) -> pd.DataFrame:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        delimiters = [',', ';', '\t']
        for enc in encodings:
            for delim in delimiters:
                try:
                    df = pd.read_csv(caminho_arquivo, encoding=enc, delimiter=delim)
                    if not df.empty:
                        return df
                except Exception:
                    continue
        print(f"[-] Erro: Não foi possível ler o CSV '{caminho_arquivo}' com as codificações/delimitadores tentados.")
        return pd.DataFrame()

    def _ler_excel(self, caminho_arquivo: str) -> pd.DataFrame:
        try:
            return pd.read_excel(caminho_arquivo)
        except Exception as e:
            print(f"[-] Erro ao ler Excel: {e}")
            return pd.DataFrame()

    def _processar_dataframe_para_chunks(self, df: pd.DataFrame, nome_arquivo: str) -> List[str]:
        chunks = []
        if df.empty:
            return chunks
        col_pergunta = next((col for col in df.columns if 'pergunta' in str(col).lower()), None)
        col_resposta = next((col for col in df.columns if 'resposta' in str(col).lower()), None)
        if col_pergunta and col_resposta:
            for index, row in df.iterrows():
                pergunta = str(row[col_pergunta]).strip()
                resposta = str(row[col_resposta]).strip()
                if pergunta and resposta:
                    chunks.append(f"Pergunta: {pergunta}\nResposta: {resposta}")
        else:
            print(f"[!] Atenção: Não foram encontradas colunas 'Pergunta'/'Resposta' em '{nome_arquivo}'. Processando linhas como chunks.")
            for index, row in df.iterrows():
                chunk_data = " ".join([str(val) for val in row.values if pd.notna(val) and str(val).strip()])
                if chunk_data:
                    chunks.append(chunk_data)
        return [chunk for chunk in chunks if chunk.strip()]

    def adicionar_documento(self, caminho_arquivo: str) -> str:
        nome_arquivo = os.path.basename(caminho_arquivo)
        extensao = os.path.splitext(nome_arquivo)[1].lower()
        tema = os.path.basename(os.path.dirname(caminho_arquivo))
        conteudo_texto_ou_df = None
        print(f"[i] Processando documento: {nome_arquivo} (Tipo: {extensao}, Tema: {tema})")
        if extensao == '.pdf':
            conteudo_texto_ou_df = self._ler_pdf(caminho_arquivo)
        elif extensao == '.txt':
            conteudo_texto_ou_df = self._ler_txt(caminho_arquivo)
        elif extensao == '.docx':
            conteudo_texto_ou_df = self._ler_docx(caminho_arquivo)
        elif extensao in ['.csv']:
            conteudo_texto_ou_df = self._ler_csv(caminho_arquivo)
        elif extensao in ['.xlsx', '.xls']:
            conteudo_texto_ou_df = self._ler_excel(caminho_arquivo)
        elif extensao in ['.jpg', '.jpeg', '.png', '.webp']:
            if OneRing.MOTOR_IA == 'gemini':
                _instrucao_img = "Haja como um especialista em analisar e descrever imagens"
                _contexto_img = "Otimize a descrição da imagem para um sistema RAG resgatar informações para uma IA analisar e responder ao usuário, incluindo no texto uma sessão de possíveis perguntas e respostas sobre a imagem  ou sobre o assunto relacionado à imagem."
                _comando_img = "Descreva a imagem com detalhes, incluindo cores, pessoas, animais, formas, objetos e contexto."
                conteudo_texto_ou_df = Gemma_IA_API.consultar_gemma_api_gemini(_instrucao_img, _contexto_img, _comando_img, caminho_arquivo,"gemma-3-27b-it")
            else:
                _instrucao_img = "Haja como um especialista em analisar e descrever imagens"
                _contexto_img = "Otimize a descrição da imagem para um sistema RAG resgatar informações para uma IA analisar e responder ao usuário, incluindo no texto uma sessão de possíveis perguntas e respostas sobre a imagem ou sobre o assunto relacionado à imagem, como o que é o objeto foco da imagem."
                _comando_img = "Descreva a imagem com detalhes, incluindo cores, pessoas, animais, formas, objetos e contexto."
                conteudo_texto_ou_df = Gemma_IA.consultar_ollama_local(_instrucao_img, _contexto_img, _comando_img, caminho_arquivo,"gemma3:4b")
            print(f"  [+] Descrição da imagem recebida: \n{conteudo_texto_ou_df}")
        else:
            print(f"[-] Tipo de arquivo '{extensao}' não suportado.")
            return None
        chunks = []
        if isinstance(conteudo_texto_ou_df, str):
            if conteudo_texto_ou_df.strip():
                chunks = self._dividir_texto_em_chunks(conteudo_texto_ou_df)
        elif isinstance(conteudo_texto_ou_df, pd.DataFrame):
            chunks = self._processar_dataframe_para_chunks(conteudo_texto_ou_df, nome_arquivo)
        if not chunks:
            print(f"[-] Conteúdo vazio ou erro no processamento do arquivo '{nome_arquivo}'. Não adicionado.")
            return None
        nome_nova_colecao = self._obter_proximo_nome_colecao()
        try:
            colecao = self.client.get_or_create_collection(
                name=nome_nova_colecao,
                embedding_function=embedding_function
            )
        except Exception as e:
            print(f"[-] Erro ao obter/criar coleção '{nome_nova_colecao}': {e}")
            return None
        documents = []
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{nome_nova_colecao}_chunk_{i}_{uuid.uuid4().hex[:8]}"
            documents.append(chunk)
            metadatas.append({
                "id_colecao": nome_nova_colecao,
                "nome_arquivo_original": nome_arquivo,
                "extensao": extensao,
                "caminho_completo": caminho_arquivo,
                "numero_chunk": i,
                "tipo_conteudo": "texto_extraido" if extensao not in ['.jpg', '.jpeg', '.png', '.webp'] else "descricao_ia_imagem",
                "tema": tema
            })
            ids.append(chunk_id)
        try:
            colecao.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self.lista_nomes_colecoes.append(nome_nova_colecao)
            if tema not in self.temas_disponiveis:
                self.temas_disponiveis.append(tema)
            self._salvar_lista_colecoes()
            print(f"[+] Documento '{nome_arquivo}' processado e adicionado como coleção '{nome_nova_colecao}'.")
            return nome_nova_colecao
        except Exception as e:
            print(f"[-] Erro ao adicionar chunks ao ChromaDB para '{nome_arquivo}': {e}")
            return None

    def deletar_colecao_por_nome(self, nome_colecao: str) -> bool:
        """Deleta uma coleção específica pelo nome."""
        if nome_colecao not in self.lista_nomes_colecoes:
            print(f"[-] Coleção '{nome_colecao}' não encontrada na lista de coleções registradas.")
            return False
        try:
            self.client.delete_collection(name=nome_colecao)
            self.lista_nomes_colecoes.remove(nome_colecao)
            self.temas_disponiveis = self._carregar_temas_disponiveis()
            self._salvar_lista_colecoes()
            print(f"[+] Coleção '{nome_colecao}' deletada com sucesso.")
            return True
        except Exception as e:
            print(f"[-] Erro ao deletar coleção '{nome_colecao}': {e}")
            return False

    def zerar_todas_colecoes(self):
        print(f"[i] Iniciando o processo de zerar todas as coleções.")
        
        # Obtém a lista atualizada de coleções diretamente do servidor
        try:
            colecoes_no_servidor = self.client.list_collections()
            if not colecoes_no_servidor:
                print("[!] Nenhuma coleção encontrada no servidor para zerar.")
                self.lista_nomes_colecoes = []
                self.temas_disponiveis = []
                self._salvar_lista_colecoes()
                return
        except Exception as e:
            print(f"[-] Erro ao listar coleções do servidor: {e}")
            return

        print(f"[i] Encontradas {len(colecoes_no_servidor)} coleções para deletar...")
        
        sucesso = True
        for colecao in colecoes_no_servidor:
            try:
                print(f"  - Deletando coleção '{colecao.name}'...")
                self.client.delete_collection(name=colecao.name)
            except Exception as e:
                print(f"  [-] Falha ao deletar a coleção '{colecao.name}': {e}")
                sucesso = False

        if sucesso:
            print("[v] Todas as coleções foram removidas do servidor com sucesso!")
        else:
            print("[!] Algumas coleções podem não ter sido removidas. Verifique os logs.")

        # Limpa as listas locais e salva o estado vazio
        self.lista_nomes_colecoes = []
        self.temas_disponiveis = []
        self._salvar_lista_colecoes()

    def criar_colecoes_da_pasta(self, pasta_documentos: str = PERSIST_PASTA_BIBLIOTECA):
        print(f"\n--- [i] Criando/Atualizando coleções da pasta '{pasta_documentos}' ---")
        os.makedirs(pasta_documentos, exist_ok=True)
        arquivos_na_pasta = [f for f in os.listdir(pasta_documentos) if os.path.isfile(os.path.join(pasta_documentos, f))]
        if not arquivos_na_pasta:
            print(f"[!] Nenhum arquivo encontrado em '{pasta_documentos}'.")
            return
        for arquivo in arquivos_na_pasta:
            if arquivo.lower() == ".gitkeep":
                continue
            caminho_completo_arquivo = os.path.join(pasta_documentos, arquivo)
            self.adicionar_documento(caminho_completo_arquivo)
        print("\n--- [i] Processamento de documentos da pasta concluído. ---")
        print(f"[+] Total de coleções ativas no ChromaDB: {len(self.lista_nomes_colecoes)}")
        print(f"[+] Total de chunks no ChromaDB: {self.total_chunks_no_bd()}")
    
    def processar_uploads_e_criar_colecoes(self, files: List, tema: str) -> dict:
        temp_base_dir = "temp_uploads"
        request_id = str(uuid.uuid4())
        unique_request_dir = os.path.join(temp_base_dir, request_id)
        diretorio_processamento = os.path.join(unique_request_dir, tema)
        try:
            os.makedirs(diretorio_processamento, exist_ok=True)
            print(f"[i] Diretório temporário de processamento criado: {diretorio_processamento}")
            if not files: return {"sucesso": False, "mensagem": "Nenhum arquivo foi enviado."}
            nomes_arquivos = []
            for file in files:
                if file and file.filename:
                    caminho_arquivo_salvo = os.path.join(diretorio_processamento, file.filename)
                    file.save(caminho_arquivo_salvo)
                    nomes_arquivos.append(file.filename)
            if not nomes_arquivos: return {"sucesso": False, "mensagem": "Os arquivos enviados são inválidos ou não têm nome."}
            print(f"[i] {len(nomes_arquivos)} arquivos salvos em pasta temporária: {', '.join(nomes_arquivos)}")
            print(f"[i] Chamando 'criar_colecoes_da_pasta' para o tema '{tema}'...")
            self.criar_colecoes_da_pasta(pasta_documentos=diretorio_processamento)
            mensagem_sucesso = f"{len(nomes_arquivos)} arquivos do tema '{tema}' foram processados e adicionados com sucesso."
            print(f"[+] {mensagem_sucesso}")
            return {"sucesso": True, "mensagem": mensagem_sucesso}
        except Exception as e:
            mensagem_erro = f"Ocorreu um erro inesperado ao processar os arquivos para o tema '{tema}': {e}"
            print(f"[-] {mensagem_erro}")
            return {"sucesso": False, "mensagem": mensagem_erro}
        finally:
            if os.path.exists(unique_request_dir):
                print(f"[i] Limpando diretório temporário completo: {unique_request_dir}")
                shutil.rmtree(unique_request_dir)
                print("[+] Limpeza do diretório temporário concluída.")

    def listar_colecoes(self) -> List[dict]:
        """
        Lista todas as coleções existentes, ordenadas por tema, nome do arquivo e ID,
        e retorna a lista formatada.
        """
        try:
            server_collections = self.client.list_collections()
        except Exception as e:
            print(f"[-] Erro ao conectar ou listar coleções do servidor: {e}")
            return []

        self.lista_nomes_colecoes = [c.name for c in server_collections]
        self._salvar_lista_colecoes()

        if not self.lista_nomes_colecoes:
            print("[!] Nenhuma coleção encontrada no servidor.")
            return []

        colecoes_para_ordenar = []
        for nome_colecao in self.lista_nomes_colecoes:
            try:
                colecao = self.client.get_collection(name=nome_colecao, embedding_function=embedding_function)
                metadados_sample = colecao.peek(limit=1)['metadatas']
                
                if not metadados_sample:
                    # Adiciona com valores padrão para garantir a ordenação
                    colecoes_para_ordenar.append({
                        "id": nome_colecao, 
                        "nome_arquivo": "N/A", 
                        "tema": "Sem Tema"
                    })
                    continue
                
                meta = metadados_sample[0]
                nome_arquivo = meta.get("nome_arquivo_original", "N/A")
                tema = meta.get("tema", "Sem Tema")
                
                colecoes_para_ordenar.append({
                    "id": nome_colecao,
                    "nome_arquivo": nome_arquivo,
                    "tema": tema
                })
            except Exception as e:
                print(f"[-] Erro ao acessar a coleção '{nome_colecao}': {e}")

        # --- A MÁGICA DA ORDENAÇÃO ACONTECE AQUI ---
        # Ordena a lista usando uma tupla como chave: (tema, nome_arquivo, id)
        # str.lower é usado para garantir que a ordenação não diferencie maiúsculas de minúsculas
        colecoes_formatadas = sorted(
            colecoes_para_ordenar, 
            key=lambda x: (
                str(x['tema']).lower(), 
                str(x['nome_arquivo']).lower(), 
                str(x['id']).lower()
            )
        )

        # Imprime a lista já ordenada no terminal
        print("\n--- LISTA DE COLEÇÕES EXISTENTES (ORDENADA) ---")
        for item in colecoes_formatadas:
            print(f"- {item['id']} | {item['nome_arquivo']} | {item['tema']}")
        print("---------------------------------------------")

        return colecoes_formatadas
    
    def total_chunks_no_bd(self) -> int:
        total = 0
        colecoes_servidor = self.client.list_collections()
        for colecao in colecoes_servidor:
            try:
                total += colecao.count()
            except Exception as e:
                print(f"[!] Atenção: Não foi possível contar chunks para coleção '{colecao.name}': {e}.")
        return total

    def consultar_multiplas_colecoes(self, 
                                     pergunta: str, 
                                     usar_ia_local: bool = False, 
                                     instrucao: str = "", 
                                     pdf_path: str = None, 
                                     imagem_path: str = None, 
                                     modelo_de_pensamento: str = "gemma-3-27b-it", 
                                     n_results_per_colecao: int = 10, 
                                     max_distance_threshold: float = 0.8) -> str:
        print(f"\n--- [i] Iniciando consulta RAG para: '{pergunta}' (IA: {'LOCAL' if usar_ia_local else 'API'}) ---")
        
        # 1. Classifica a pergunta para determinar o tema
        tema_usuario = self._classificar_pergunta_por_tema(pergunta)
        if not tema_usuario:
            return "Não foi possível identificar o tema da pergunta. Temas disponíveis: " + ", ".join(self.temas_disponiveis)
        
        print(f"\nA pergunta é: '{pergunta}'.")
        print(f"O tema identificado para a pergunta é: '{tema_usuario}'.")

        # 2. Filtra os nomes das coleções pelo tema identificado
        colecoes_filtradas_nomes = []
        for nome_colecao in self.lista_nomes_colecoes:
            try:
                # Obtém a coleção para inspecionar seus metadados
                colecao = self.client.get_collection(name=nome_colecao, embedding_function=embedding_function)
                # Pega o metadado do primeiro item como amostra do tema da coleção
                metadata_sample = colecao.peek(limit=1)['metadatas']
                if metadata_sample and metadata_sample[0].get("tema") == tema_usuario:
                    colecoes_filtradas_nomes.append(nome_colecao)
            except Exception as e:
                print(f"[-] Aviso: Erro ao verificar tema da coleção '{nome_colecao}': {e}")

        if not colecoes_filtradas_nomes:
            return f"Nenhuma coleção encontrada para o tema '{tema_usuario}'."

        print(f"[i] Consultando {len(colecoes_filtradas_nomes)} coleções do tema '{tema_usuario}'...")

        # 3. Itera sobre as coleções filtradas e faz a consulta em cada uma
        contextos_relevantes = []
        for nome_colecao in colecoes_filtradas_nomes:
            try:
                # Obtém o objeto da coleção
                colecao_chroma = self.client.get_collection(name=nome_colecao, embedding_function=embedding_function)
                if colecao_chroma.count() == 0:
                    continue
                
                # CHAMA O .QUERY() NO OBJETO DA COLEÇÃO (O JEITO CORRETO)
                resultados = colecao_chroma.query(
                    query_texts=[pergunta],
                    n_results=n_results_per_colecao,
                    include=['documents', 'metadatas', 'distances']
                )

                if resultados and resultados['documents'] and resultados['documents'][0]:
                    for i in range(len(resultados['documents'][0])):
                        distance = resultados['distances'][0][i]
                        if distance <= max_distance_threshold:
                            contextos_relevantes.append({
                                "conteudo": resultados['documents'][0][i],
                                "metadados": resultados['metadatas'][0][i],
                                "distancia": distance
                            })
            except Exception as e:
                print(f"[-] Erro ao consultar a coleção '{nome_colecao}': {e}")
                continue

        if not contextos_relevantes:
            return "Não foram encontrados documentos relevantes no Grimório para sua consulta com o tema identificado."

        # Ordena todos os resultados coletados pela distância (relevância)
        contextos_relevantes_filtrados = sorted(contextos_relevantes, key=lambda x: x['distancia'])
        
        print(f"  [+] Total de contextos relevantes encontrados: {len(contextos_relevantes_filtrados)}")

        # Monta o prompt para a IA
        instrucao_tratada = f"Instrução: {instrucao}\n" if instrucao else ""
        instrucao_tratada += " Responda estritamente com base no contexto fornecido. Se a resposta não puder ser inferida do contexto, diga que não tem informações suficientes para responder."
        contexto_tratado = "\nContexto:\n" + "\n\n".join([item['conteudo'] for item in contextos_relevantes_filtrados])
        pergunta_tratada = f"\nPergunta: {pergunta}"
        
        Canivete.salvar_txt(instrucao_tratada + contexto_tratado + pergunta_tratada, "prompt_completo.txt")

        # Chama a IA para gerar a resposta final
        try:
            if usar_ia_local:
                resposta_gemini = Gemma_IA.consultar_ollama_local(instrucao_tratada, contexto_tratado, pergunta_tratada, imagem_path, modelo_de_pensamento)
            else:
                resposta_gemini = Gemma_IA_API.consultar_gemma_api_gemini(instrucao_tratada, contexto_tratado, pergunta_tratada, imagem_path, modelo_de_pensamento)
        except Exception as e:
            print(f"  [-] Erro ao chamar a IA: {e}")
            return f"Ocorreu um erro ao gerar a resposta pela IA: {e}"

        print(f" [i] Método consultar_multiplas_colecoes finalizado.")
        return resposta_gemini

    def _classificar_pergunta_por_tema(self, pergunta: str) -> str:
        """Classifica a pergunta do usuário para determinar o tema."""
        if not self.temas_disponiveis:
            print("[!] Nenhum tema disponível para classificação.")
            return None

        instrucao = "Analise a pergunta do usuário e o contexto com os temas disponíveis. Sua única tarefa é retornar o nome exato de um dos temas listados que seja mais relevante para a pergunta. Responda apenas com o nome do tema e nada mais."
        contexto = f"Temas disponíveis: {', '.join(self.temas_disponiveis)}"
        
        if OneRing.PESQUISA_TEMA_IA_LOCAL:
            resposta = Gemma_IA.consultar_ollama_local(instrucao, contexto, pergunta, None, "gemma3n:latest")
        else:
            resposta = Gemma_IA_API.consultar_gemma_api_gemini(instrucao, contexto, pergunta, None, "gemma-3-27b-it")
        
        # Limpa e valida a resposta da IA
        tema_identificado = resposta.strip().lower().replace(".", "")
        
        # Busca por correspondência exata ou parcial na lista de temas
        for tema in self.temas_disponiveis:
            if tema.lower() == tema_identificado:
                return tema # Retorna o tema com a capitalização original
        
        print(f"[-] A IA retornou um tema ('{tema_identificado}') que não corresponde a nenhum tema existente.")
        return None

# ---------------------------------------------------------------------------------------------------------------------------
# Bloco de Teste Principal
# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OneRing.PASTA_BIBLIOTECA, exist_ok=True)
    with open(os.path.join(OneRing.PASTA_BIBLIOTECA, ".gitkeep"), "w") as f:
        f.write("")
    
    # A instanciação agora usa os padrões definidos no início do arquivo
    sistema_rag = SistemaRAG()
    
    while True:
        print("\n--- MENU RAG ---")
        print("1. Adicionar/Atualizar documentos na biblioteca")
        print("2. Consultar biblioteca (RAG)")
        print("3. Deletar uma coleção (documento)")
        print("4. Zerar todas as coleções")
        print("5. Listar coleções existentes")
        print("6. Sair")
        escolha = input("Escolha uma opção: ")
        if escolha == '1':
            print(f"\n--- ADICIONAR/ATUALIZAR DOCUMENTOS ---")
            print(f"Por favor, coloque seus documentos na pasta apropriada...")
            print("Exemplo: coloque arquivos sobre 'capivaras' em 'biblioteca_geral/capivaras/'")
            interacao_usuario_diretorio_teste = input("\nInforme o caminho para o diretório: \n -> ")
            input("Pressione Enter para continuar depois de colocar os arquivos...")
            sistema_rag.criar_colecoes_da_pasta(pasta_documentos=interacao_usuario_diretorio_teste)
            # A lógica precisa varrer subpastas para encontrar os temas
            # for tema_dir in os.listdir(OneRing.PASTA_BIBLIOTECA):
            #     caminho_tema = os.path.join(OneRing.PASTA_BIBLIOTECA, tema_dir)
            #     if os.path.isdir(caminho_tema):
            #         sistema_rag.criar_colecoes_da_pasta(pasta_documentos=caminho_tema)
            print(f"\n[+] Total de chunks no BD após adicionar: {sistema_rag.total_chunks_no_bd()}")
            print("\n--- ADIÇÃO DE DOCUMENTOS CONCLUÍDA ---")
        elif escolha == '2':
            # ... (código do menu de consulta sem alterações) ...
            print("\n--- CONSULTAR RAG ---")
            while True:
                try:
                    interacao_usuario = input("\nFaça sua pergunta (ou 'sair' para voltar ao menu): \n -> ")
                    if interacao_usuario.lower() == 'sair': break
                    interacao_tipo_ia = input("Usar IA Local? (s/n): ").lower()
                    usar_ia_local_flag = True if interacao_tipo_ia == 's' else False
                    interacao_usuario_modelo_ia = input("Digite o nome do modelo (ex: gemma3:latest ou gemma-3-27b-it): \n -> ")
                    if not interacao_usuario_modelo_ia:
                        interacao_usuario_modelo_ia = "gemma-3-27b-it" if not usar_ia_local_flag else "gemma3:latest"
                    interacao_usuario_quant_resultados = input("Quantos resultados (chunks) deseja buscar? (padrão: 10): \n -> ")
                    if not interacao_usuario_quant_resultados.isdigit(): interacao_usuario_quant_resultados = "10"
                    interacao_usuario_similaridade = input("Similaridade da busca RAG (0.0 a 1.0; padrão: 0.8): \n -> ")
                    try:
                        similaridade = float(interacao_usuario_similaridade)
                        if not (0.0 <= similaridade <= 1.0): raise ValueError
                    except ValueError:
                        print("[!] Valor de similaridade inválido. Usando padrão: 0.8")
                        similaridade = 0.8
                    resposta = sistema_rag.consultar_multiplas_colecoes(
                        pergunta=interacao_usuario,
                        usar_ia_local=usar_ia_local_flag,
                        instrucao="Haja como um especialista nos assuntos questionados e responda de forma clara, detalhada e didática.",
                        modelo_de_pensamento=interacao_usuario_modelo_ia,
                        n_results_per_colecao=int(interacao_usuario_quant_resultados),
                        max_distance_threshold=similaridade
                    )
                    print(f"\n--- Resposta --- \n{resposta}\n---------------")
                except KeyboardInterrupt:
                    print("\nConsulta interrompida pelo usuário.")
                    break
                except Exception as e:
                    print(f"\n[!] Ocorreu um erro durante a consulta: {e}")
        elif escolha == '3':
            print("\n--- DELETAR COLEÇÃO ---")
            sistema_rag.listar_colecoes()
            if sistema_rag.lista_nomes_colecoes:
                id_para_deletar = input("Digite o ID (nome) da coleção que deseja deletar: ").strip()
                if id_para_deletar:
                    sistema_rag.deletar_colecao_por_nome(id_para_deletar)
        elif escolha == '4':
            confirmacao = input("Tem certeza que deseja ZERAR TODAS as coleções do banco de dados? (s/n): ").lower()
            if confirmacao == 's':
                sistema_rag.zerar_todas_colecoes()
            else:
                print("Operação cancelada.")
        elif escolha == '5':
            sistema_rag.listar_colecoes()
        elif escolha == '6':
            print("Saindo do programa. Até mais!")
            break
        else:
            print("[!] Opção inválida. Por favor, escolha um número de 1 a 6.")