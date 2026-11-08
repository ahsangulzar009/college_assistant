from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

pdf_files = ["academics_handbook.pdf", "fee_structure.pdf"]

docs = []

for pdf_path in pdf_files:
  reader = PdfReader(pdf_path)
  for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if text.strip():
      docs.append(
        Document(
          page_content = text,
          metadata = {"source": pdf_path, "page": i}
        )
      )


splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap= 150
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
  documents = chunks,
  embedding = embedding_model,
  persist_directory = "chroma_db"
)
