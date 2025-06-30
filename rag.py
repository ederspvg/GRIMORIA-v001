import parametros_globais as OneRing
import utilitarios as Canivete
import prompts_ia as Persona
import uuid  # Para gerar IDs únicos para os chunks
import ia_gemma as Gemma_IA
import ia_gemma_api_gemini as Gemma_IA_API  # Adicionado para corresponder ao rag.txt original
import os
import json  # Para persistir a lista de coleções
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import chromadb
from chromadb.utils import embedding_functions  # (mantido como estava no rag.txt fornecido, embora comentado no original)
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PIL import Image
from typing import List, Tuple
import shutil
import pdfplumber
# ---AQUI---
import pandas as pd  # Adicionado para ler Excel e CSV
import csv  # Adicionado, embora pandas seja o principal para ambos

# Configuração da API Gemini
load_dotenv(dotenv_path='ambiente.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Configurações do ChromaDB
PERSIST_DIRECTORY = OneRing.PASTA_BANCO  # "chroma_db_v13"
PERSIST_PASTA_BIBLIOTECA = OneRing.PASTA_BIBLIOTECA  # "biblioteca_geral"

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
    def __init__(self, persist_directory: str = PERSIST_DIRECTORY):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.lista_colecoes_file = os.path.join(persist_directory, "lista_colecoes.json")  # Novo atributo da classe
        # Garante que a pasta do banco de dados exista
        os.makedirs(self.persist_directory, exist_ok=True)
        # Carregar a lista de nomes de coleções existentes
        self.lista_nomes_colecoes = self._carregar_lista_colecoes()
        print(f"[i] SistemaRAG inicializado. Banco de dados em: {self.persist_directory}")
        print(f"[i] {len(self.lista_nomes_colecoes)} coleções de documentos registradas.")

    def _carregar_lista_colecoes(self) -> List[str]:
        if os.path.exists(self.lista_colecoes_file):  # Usa o novo atributo
            try:
                with open(self.lista_colecoes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[-] Erro: Arquivo '{self.lista_colecoes_file}' corrompido ou vazio. Iniciando com lista vazia.")
                return []
        return []

    def _salvar_lista_colecoes(self):
        with open(self.lista_colecoes_file, 'w', encoding='utf-8') as f:  # Usa o novo atributo
            json.dump(self.lista_nomes_colecoes, f, indent=4)

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
            return str(ult_num + 1).zfill(4)  # Garante 4 dígitos, ex: 0001, 0002
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
        conteudo_texto_ou_df = None
        print(f"[i] Processando documento: {nome_arquivo} (Tipo: {extensao})")
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
                _instrucao_img = "Haja como um pintor e desenhista especialista em analisar e descrever imagens"
                _contexto_img = "Otimize a descrição para um sistema RAG resgatar informações para uma IA analisar e responder ao usuário, incluindo no texto uma sessão de possíveis perguntas e respostas sobre a imagem."
                _comando_img = "Descreva a imagem com detalhes, incluindo cores, pessoas, animais, formas, objetos e contexto."
                conteudo_texto_ou_df = Gemma_IA_API.consultar_gemma_api_gemini(_instrucao_img, _contexto_img, _comando_img, caminho_arquivo)
            else:
                _instrucao_img = "Haja como um pintor e desenhista especialista em analisar e descrever imagens"
                _contexto_img = "Otimize a descrição para um sistema RAG resgatar informações para uma IA analisar e responder ao usuário, incluindo no texto uma sessão de possíveis perguntas e respostas sobre a imagem."
                _comando_img = "Descreva a imagem com detalhes, incluindo cores, pessoas, animais, formas, objetos e contexto."
                conteudo_texto_ou_df = Gemma_IA.consultar_ollama_local(_instrucao_img, _contexto_img, _comando_img, caminho_arquivo)
            print(f"  [+] Descrição da imagem recebida: \n {conteudo_texto_ou_df}")
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
                "tipo_conteudo": "texto_extraido" if extensao not in ['.jpg', '.jpeg', '.png', '.webp'] else "descricao_ia_imagem"
            })
            ids.append(chunk_id)
        try:
            colecao.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self.lista_nomes_colecoes.append(nome_nova_colecao)
            self._salvar_lista_colecoes()
            print(f"[+] Documento '{nome_arquivo}' processado e adicionado como coleção '{nome_nova_colecao}'.")
            return nome_nova_colecao
        except Exception as e:
            print(f"[-] Erro ao adicionar chunks ao ChromaDB para '{nome_arquivo}': {e}")
            return None

    def deletar_colecao_por_nome(self, nome_colecao: str) -> bool:
        if nome_colecao not in self.lista_nomes_colecoes:
            print(f"[-] Coleção '{nome_colecao}' não encontrada na lista de coleções registradas.")
            return False
        try:
            self.client.delete_collection(name=nome_colecao)
            self.lista_nomes_colecoes.remove(nome_colecao)
            self._salvar_lista_colecoes()
            print(f"[+] Coleção '{nome_colecao}' deletada com sucesso.")
            return True
        except Exception as e:
            print(f"[-] Erro ao deletar coleção '{nome_colecao}': {e}")
            return False

    def zerar_todas_colecoes(self):
        print(f"[i] Iniciando o processo de zerar todas as coleções no diretório: {self.persist_directory}")
        colecoes_para_deletar = self.lista_nomes_colecoes[:]
        if not colecoes_para_deletar:
            print("[!] Nenhuma coleção encontrada para zerar.")
            print("[v] Todas as coleções já estão zeradas (ou não existem).")
            if os.path.exists(self.lista_colecoes_file):  # Usa o novo atributo
                try:
                    os.remove(self.lista_colecoes_file)
                    print(f"[+] Arquivo '{self.lista_colecoes_file}' removido com sucesso.")
                except OSError as e:
                    print(f"[-] Erro ao remover '{self.lista_colecoes_file}': {e}")
            self.lista_nomes_colecoes = []
            return
        try:
            for nome_colecao in colecoes_para_deletar:
                try:
                    self.client.delete_collection(name=nome_colecao)
                    print(f"[+] Coleção '{nome_colecao}' deletada com sucesso.")
                except Exception as e:
                    print(f"[-] Erro ao deletar a coleção '{nome_colecao}': {e}")
            self.lista_nomes_colecoes = []
            self._salvar_lista_colecoes()
            print("[+] Lista de coleções em memória e arquivo 'lista_colecoes.json' zerados.")
            print("[v] Todas as coleções foram zeradas com sucesso!")
        except Exception as e:
            print(f"[-] Ocorreu um erro inesperado ao zerar coleções: {e}")
            print("Verifique se o diretório do banco de dados não está sendo acessado por outros processos.")

    def criar_colecoes_da_pasta(self, pasta_documentos: str = PERSIST_PASTA_BIBLIOTECA):
        print(f"\n--- [i] Criando/Atualizando coleções da pasta '{pasta_documentos}' ---")
        os.makedirs(pasta_documentos, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.zerar_todas_colecoes()
        arquivos_na_pasta = [f for f in os.listdir(pasta_documentos) if os.path.isfile(os.path.join(pasta_documentos, f))]
        if not arquivos_na_pasta:
            print(f"[!] Nenhum arquivo encontrado em '{pasta_documentos}'.")
            return
        for arquivo in arquivos_na_pasta:
            if arquivo.lower() == ".gitkeep":
                print(f"  [i] Ignorando arquivo .gitkeep: {arquivo}")
                continue
            caminho_completo_arquivo = os.path.join(pasta_documentos, arquivo)
            self.adicionar_documento(caminho_completo_arquivo)
        print("\n--- [i] Processamento de documentos da pasta concluído. ---")
        print(f"[+] Total de coleções ativas no ChromaDB: {len(self.lista_nomes_colecoes)}")
        print(f"[+] Total de chunks no ChromaDB: {self.total_chunks_no_bd()}")

    def total_chunks_no_bd(self) -> int:
        total = 0
        for nome_colecao in self.lista_nomes_colecoes:
            try:
                colecao = self.client.get_collection(name=nome_colecao, embedding_function=embedding_function)
                total += colecao.count()
            except Exception as e:
                print(f"[!] Atenção: Não foi possível contar chunks para coleção '{nome_colecao}': {e}. Pode estar corrompida ou ausente.")
        return total

    def consultar_multiplas_colecoes(self, 
                                     pergunta: str, 
                                     usar_ia_local: bool = False, 
                                     instrucao: str = "", 
                                     pdf_path: str = None, 
                                     imagem_path: str = None, 
                                     modelo_de_pensamento: str = "gemini-1.5-flash-latest", 
                                     n_results_per_colecao: int = 10, 
                                     max_distance_threshold: float = 0.8) -> str:
        print(f"\n--- [i] Iniciando consulta RAG para: '{pergunta}' (IA: {'LOCAL' if usar_ia_local else 'API'}) ---")
        contextos_relevantes = []
        for nome_colecao in self.lista_nomes_colecoes:
            try:
                colecao_chroma = self.client.get_collection(name=nome_colecao, embedding_function=embedding_function)
                if colecao_chroma.count() == 0:
                    continue
                resultados = colecao_chroma.query(
                    query_texts=[pergunta],
                    n_results=n_results_per_colecao,
                    include=['documents', 'metadatas', 'distances']
                )
                if resultados and resultados['documents'] and resultados['documents'][0]:
                    for i in range(len(resultados['documents'][0])):
                        doc_content = resultados['documents'][0][i]
                        meta = resultados['metadatas'][0][i]
                        distance = resultados['distances'][0][i]
                        if distance <= max_distance_threshold:
                            contextos_relevantes.append({
                                "conteudo": doc_content,
                                "metadados": meta,
                                "distancia": distance
                            })
            except Exception as e:
                print(f"[-] Erro ao consultar coleção '{nome_colecao}': {e}")
                continue
        contextos_relevantes_filtrados = sorted(contextos_relevantes, key=lambda x: x['distancia'])
        print(f"  [+ Contextos relevantes encontrados: {len(contextos_relevantes_filtrados)}")
        if not contextos_relevantes_filtrados:
            return "Não foram encontrados documentos relevantes no Grimório para sua consulta."
        # prompt_final = f"Instrução: {instrucao}\n" if instrucao else ""
        # prompt_final += "Contexto dos documentos:\n" + "\n".join([item['conteudo'] for item in contextos_relevantes_filtrados])
        # prompt_final += f"\nPergunta: {pergunta}"
        # prompt_final += "\nResponda estritamente com base no contexto fornecido. Se a resposta não puder ser inferida do contexto, diga que não tem informações suficientes para responder."
        
        instrucao_tratada   = f"Instrução: {instrucao}\n" if instrucao else ""
        instrucao_tratada   += "\nResponda estritamente com base no contexto fornecido. Se a resposta não puder ser inferida do contexto, diga que não tem informações suficientes para responder."
        contexto_tratado    = "Contexto:\n" + "\n".join([item['conteudo'] for item in contextos_relevantes_filtrados])
        pergunta_tratada    = f"\nPergunta: {pergunta}"
        
        try:
            if usar_ia_local:
                print(f"  [i] Chamando Gemma_IA.consultar_gemma_local para gerar resposta...")
                # resposta_gemini = Gemma_IA.consultar_gemma_local(prompt_final, "")
                resposta_gemini = Gemma_IA.consultar_ollama_local(
                    instrucao_tratada,
                    contexto_tratado,
                    pergunta_tratada,
                    imagem_path
                )
                print(f"  [+] Resposta do Gemma LOCAL recebida.")
            else:
                print(f"  [i] Chamando API Gemini para gerar resposta...")
                # model = genai.GenerativeModel(modelo_de_pensamento)
                # resposta_gemini = model.generate_content(prompt_final).text
                resposta_gemini = Gemma_IA_API.consultar_gemma_api_gemini(
                    instrucao_tratada,
                    contexto_tratado,
                    pergunta_tratada,
                    imagem_path
                )
                print(f"  [+] Resposta do Gemini API recebida.")
        except Exception as e:
            print(f"  [-] Erro ao chamar a IA: {e}")
            error_message = str(e)
            if hasattr(e, 'message'): error_message = e.message
            if "block_reason: SAFETY" in error_message or "response was blocked" in error_message:
                resposta_gemini = "A resposta foi bloqueada devido às configurações de segurança."
            elif "token" in error_message.lower():
                resposta_gemini = f"Ocorreu um erro relacionado ao limite de tokens: {error_message}"
            else:
                resposta_gemini = f"Ocorreu um erro ao gerar a resposta pela IA: {error_message}"
        print(f" [i] Método consultar_multiplas_colecoes finalizado.")
        return resposta_gemini


# ---------------------------------------------------------------------------------------------------------------------------
# Bloco de Teste Principal
# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OneRing.PASTA_BANCO, exist_ok=True)
    os.makedirs(OneRing.PASTA_BIBLIOTECA, exist_ok=True)
    with open(os.path.join(OneRing.PASTA_BIBLIOTECA, ".gitkeep"), "w") as f:
        f.write("")
    print(f"\n--- ATENÇÃO: Adicione a seguinte pasta ao seu .gitignore: ---")
    print(f"- {os.path.normpath(OneRing.PASTA_BANCO)}/")
    print(f"----------------------------------------------------------\n")
    sistema_rag = SistemaRAG(persist_directory=OneRing.PASTA_BANCO)
    while True:
        print("\n--- MENU RAG ---")
        print("1. Adicionar/Atualizar documentos na biblioteca")
        print("2. Consultar biblioteca (RAG)")
        print("3. Deletar uma coleção (documento)")
        print("4. Zerar todas as coleções")
        print("5. Sair")
        escolha = input("Escolha uma opção: ")
        if escolha == '1':
            print(f"\n--- ADICIONAR/ATUALIZAR DOCUMENTOS ---")
            print(f"Por favor, coloque seus documentos (PDF, TXT, DOCX, Imagens, CSV, XLSX) na pasta: '{OneRing.PASTA_BIBLIOTECA}'")
            input("Pressione Enter para continuar depois de colocar os arquivos...")
            sistema_rag.criar_colecoes_da_pasta(pasta_documentos=OneRing.PASTA_BIBLIOTECA)
            print(f"\n[+] Total de chunks no BD após adicionar: {sistema_rag.total_chunks_no_bd()}")
            print("\n--- ADIÇÃO DE DOCUMENTOS CONCLUÍDA ---")
        elif escolha == '2':
            print("\n--- CONSULTAR RAG ---")
            while True:
                try:
                    interacao_usuario = input("\nFaça sua pergunta (ou 'sair' para voltar ao menu): \n -> ")
                    if interacao_usuario.lower() == 'sair':
                        break
                    interacao_tipo_ia = input("Usar IA Local? (s/n): ").lower()
                    usar_ia_local_flag = True if interacao_tipo_ia == 's' else False
                    interacao_usuario_quant_resultados = input("Quantos resultados (chunks) por documento deseja buscar? (padrão: 10): \n -> ")
                    if not interacao_usuario_quant_resultados.isdigit():
                        interacao_usuario_quant_resultados = "10"
                    interacao_usuario_similaridade = input("Similaridade da busca RAG (0.0 a 1.0, 0.0=exato, 1.0=semelhante; padrão: 0.8): \n -> ")
                    try:
                        similaridade = float(interacao_usuario_similaridade)
                        if not (0.0 <= similaridade <= 1.0):
                            raise ValueError
                    except ValueError:
                        print("[!] Valor de similaridade inválido. Usando padrão: 0.8")
                        similaridade = 0.8
                    resposta = sistema_rag.consultar_multiplas_colecoes(
                        pergunta=interacao_usuario,
                        usar_ia_local=usar_ia_local_flag,
                        instrucao="Haja como um especialista nos assuntos questionados e responda de forma clara, detalhada e ao mesmo tempo didática.",
                        modelo_de_pensamento="gemma-3-27b-it",
                        n_results_per_colecao=int(interacao_usuario_quant_resultados),
                        max_distance_threshold=similaridade
                    )
                    print(f"\n--- Resposta --- \n{resposta}\n---------------")
                except KeyboardInterrupt:
                    print("\nConsulta interrompida pelo usuário.")
                    break
                except Exception as e:
                    print(f"\n[!] Ocorreu um erro durante a consulta: {e}")
            print("\n--- TESTE CONSULTAR CONCLUÍDO ---\n")
        elif escolha == '3':
            print("\n--- DELETAR COLEÇÃO ---")
            if not sistema_rag.lista_nomes_colecoes:
                print("[!] Nenhuma coleção para deletar.")
            else:
                print("Coleções existentes:")
                for i, nome_col in enumerate(sistema_rag.lista_nomes_colecoes):
                    print(f"{i+1}. {nome_col}")
                try:
                    indice_para_deletar = int(input("Digite o número da coleção que deseja deletar: ")) - 1
                    if 0 <= indice_para_deletar < len(sistema_rag.lista_nomes_colecoes):
                        nome_col_deletar = sistema_rag.lista_nomes_colecoes[indice_para_deletar]
                        sistema_rag.deletar_colecao_por_nome(nome_col_deletar)
                    else:
                        print("[!] Índice inválido.")
                except ValueError:
                    print("[!] Entrada inválida. Digite um número.")
            print("\n--- DELEÇÃO CONCLUÍDA ---")
        elif escolha == '4':
            confirmacao = input("Tem certeza que deseja zerar TODAS as coleções? (s/n): ").lower()
            if confirmacao == 's':
                sistema_rag.zerar_todas_colecoes()
            else:
                print("Operação cancelada.")
            print("\n--- ZERAR TUDO CONCLUÍDO ---")
        elif escolha == '5':
            print("Saindo do programa. Até mais!")
            break
        else:
            print("[!] Opção inválida. Por favor, escolha um número de 1 a 5.")