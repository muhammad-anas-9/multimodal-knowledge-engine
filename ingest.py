from pathlib import Path

from config import load_environment

# Load secrets before any LlamaIndex/LlamaParse imports.
load_environment(required_env_vars=("LLAMA_CLOUD_API_KEY",))

from llama_parse import LlamaParse


def parse_pdf_to_markdown(file_path: str):
    """Parse a PDF and return LlamaIndex Document objects in markdown format."""
    pdf_path = Path(file_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(str(pdf_path))
    return documents


if __name__ == "__main__":
    sample_pdf = Path("sample.pdf")
    if not sample_pdf.exists():
        raise FileNotFoundError(
            f"Test PDF missing: {sample_pdf.resolve()}. "
            "Place a PDF named 'sample.pdf' in this folder or edit ingest.py."
        )

    parsed_docs = parse_pdf_to_markdown(str(sample_pdf))
    print(f"Parsed documents: {len(parsed_docs)}")
    if parsed_docs:
        preview = parsed_docs[0].text[:500].replace("\n", " ")
        print(f"First chunk preview: {preview}")
