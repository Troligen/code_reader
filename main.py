from vectorDB.vdb import FaissIndex
from dotenv import load_dotenv
from pathFinder import find_py_files_recursively
from codeparser.parser import parse_code
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import os


load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
EXCLUDE = ["venv", ".git", ".vscode", "__pycache__", ".dontnet", ".lightning_studio",".vscode-server"]

class Query(BaseModel):
    query_text: str
    aditional_info: str


def main():
    # Initialize FAISS index manager with OpenAI embeddings
    api_key = str(OPENAI_KEY)
    faiss_manager = FaissIndex(dimension=1536, api_key=api_key)

    print("FAISS Index Manager initialized.")

    # Find all .py files in the current directory and subdirectories
    py_files_list = find_py_files_recursively(exclude_dirs=EXCLUDE)

    print("Python files found successfully.")

    # Parse the code in each file
    code_chunks = []
    for file_path in py_files_list:
        code_chunks.extend(parse_code(file_path))

    print("Code parsed successfully.")

    # Extract function and class names from the code chunks and combines the code into a single string into a new list
    text_chunks = []
    for chunk in code_chunks:
        file_name = chunk[0]
        function_name = chunk[3].split("(")[0]
        class_name = chunk[3].split("(")[0]
        line_start = chunk[1]
        line_end = chunk[2]
        code = chunk[3]

        if function_name:
            text_chunks.append(f"file_name={file_name}:class_name={function_name}:lines=Line_{line_start}-{line_end}:code_snippet={code}")
        elif class_name:
            text_chunks.append(f"file_name={file_name}:class_name={class_name}:lines=Line_{line_start}-{line_end}:code_snippet={code}")

    # Add text to the index
    try:
        for text in text_chunks:
            faiss_manager.add_text([text])
        print("Texts added to the index.")
    except AssertionError as e:
        print(f"Error adding texts: {e}")
        return

    # Search for a query text in the index
    query_text = "FaissIndex"
    results = faiss_manager.search_text(query_text, k=1)
    #print(f"Search results for '{query_text}': {results[0][1]}")

main()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/gpt-call")
def get_gpt_call(query: Query):
    client = OpenAI(api_key=api_key)
    complition = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content": "Summarize the code and explain what it does. U also need to return the code snippet that your're givven"},
            {"role":"system", "content": f"additional info: {query.aditional_info}"},
            {"role":"user", "content": query.query_text}
        ], 
        max_tokens=1000
    )
    return {"message": complition.choices[0].message["content"]}