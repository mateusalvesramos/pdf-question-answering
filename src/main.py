"""
Main entry point and interactive CLI for the PDF Question Answering system.

Orchestration flow:
1. Ingestion Phase (runs once per document):
   - Extract pages from the PDF (pdf_extractor).
   - Split pages into overlapping chunks with page tracking (chunker).
2. Inference Phase (interactive loop):
   - Prompt user for questions.
   - Retrieve top-k relevant chunks via TF-IDF (retriever).
   - Generate grounded answer via Google Gemini API (answer_generator).
   - Display response with referenced page numbers.
"""

import argparse
import sys
from pathlib import Path

from src.answer_generator import generate_answer
from src.chunker import chunk_pages
from src.pdf_extractor import (
    EmptyDocumentError,
    PDFExtractionError,
    extract_pages,
)
from src.retriever import retrieve_relevant_chunks


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PDF Question Answering with RAG (TF-IDF + Google Gemini)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=None,
        help="Path to the target PDF file. If omitted, will prompt interactively.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve for each question.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum character length per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlapping characters between consecutive chunks.",
    )
    return parser.parse_args()


def display_welcome_banner(pdf_path: Path, num_pages: int, num_chunks: int) -> None:
    """Print an informative startup summary."""
    print("\n" + "=" * 60)
    print(" PDF Question Answering (RAG com Gemini)")
    print("=" * 60)
    print(f" Arquivo carregado : {pdf_path.name}")
    print(f" Paginas extraidas : {num_pages}")
    print(f" Chunks gerados    : {num_chunks}")
    print("=" * 60)
    print(" Digite sua pergunta ou 'sair' / 'exit' para encerrar.")
    print("=" * 60 + "\n")


def interactive_qa_loop(chunks: list[dict], top_k: int) -> None:
    """Run the interactive question-answering REPL loop."""
    while True:
        try:
            query = input("\nPergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSaindo... Ate logo!")
            break

        if not query:
            continue

        if query.lower() in ("sair", "exit", "quit", "q"):
            print("Encerrando a aplicacao.")
            break

        print("\nBuscando trechos mais relevantes...")
        relevant_chunks = retrieve_relevant_chunks(
            query=query,
            chunks=chunks,
            top_k=top_k,
        )

        if not relevant_chunks:
            print("[AVISO] Nenhum trecho relevante encontrado no documento para esta pergunta.")
            continue

        print(f"-> {len(relevant_chunks)} trecho(s) recuperado(s):")
        for chunk in relevant_chunks:
            pages = ", ".join(str(p) for p in chunk["page_numbers"])
            print(f"   * Chunk #{chunk['chunk_id']} (Pag. {pages}) | Relevancia: {chunk['score']:.4f}")

        print("Gerando resposta com o Gemini...\n")
        try:
            result = generate_answer(query=query, context_chunks=relevant_chunks)
        except ValueError as err:
            print(f"[ERRO] Configuracao: {err}")
            print("Verifique se a variavel GEMINI_API_KEY esta configurada no arquivo .env\n")
            continue
        except Exception as err:
            print(f"[ERRO] Falha ao comunicar com a API do Gemini: {err}\n")
            continue

        print("-" * 60)
        print("RESPOSTA:")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)

        if result["pages_used"]:
            pages_formatted = ", ".join(str(p) for p in result["pages_used"])
            print(f"Paginas consultadas como referencia: {pages_formatted}")
        print("-" * 60)


def main() -> None:
    """Main application orchestrator."""
    args = parse_arguments()

    pdf_target = args.pdf_path
    if not pdf_target:
        pdf_target = input("Informe o caminho do arquivo PDF: ").strip()

    if not pdf_target:
        print("[ERRO] Nenhum arquivo informado. Encerrando.")
        sys.exit(1)

    pdf_path = Path(pdf_target)

    # 1. Ingestion Phase: PDF Extraction
    print(f"\n[1/2] Lendo e extraindo texto de '{pdf_path.name}'...")
    try:
        pages = extract_pages(pdf_path)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo nao encontrado: {pdf_path}")
        sys.exit(1)
    except ValueError as err:
        print(f"[ERRO] Formato invalido: {err}")
        sys.exit(1)
    except EmptyDocumentError as err:
        print(f"[AVISO] Documento sem texto extraivel: {err}")
        sys.exit(1)
    except PDFExtractionError as err:
        print(f"[ERRO] Falha ao processar o PDF: {err}")
        sys.exit(1)

    # 2. Ingestion Phase: Chunking
    print("[2/2] Fatiando conteudo em chunks...")
    chunks = chunk_pages(
        pages=pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not chunks:
        print("[AVISO] Nao foi possivel gerar chunks a partir do documento.")
        sys.exit(1)

    # 3. Inference Phase: Interactive Loop
    display_welcome_banner(pdf_path, num_pages=len(pages), num_chunks=len(chunks))
    interactive_qa_loop(chunks=chunks, top_k=args.top_k)


if __name__ == "__main__":
    main()
