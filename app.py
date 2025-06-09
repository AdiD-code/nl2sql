from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3
import os
from typing import List, Optional, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import json
from functools import lru_cache
import uuid

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_output_tokens=2048,
)

# Few-shot examples for different types of queries
FEW_SHOT_EXAMPLES = {
    "basic": [
        {
            "question": "What is the total number of records?",
            "sql": "SELECT COUNT(*) as total_records FROM data;"
        },
        {
            "question": "Show me the top 5 rows",
            "sql": "SELECT * FROM data LIMIT 5;"
        }
    ],
    "aggregation": [
        {
            "question": "What is the average value of the first numeric column?",
            "sql": "SELECT AVG(CAST(column1 AS FLOAT)) as average FROM data WHERE column1 IS NOT NULL;"
        },
        {
            "question": "What is the sum of all values in the second column?",
            "sql": "SELECT SUM(CAST(column2 AS FLOAT)) as total FROM data WHERE column2 IS NOT NULL;"
        }
    ],
    "filtering": [
        {
            "question": "Show me records where column1 is greater than 100",
            "sql": "SELECT * FROM data WHERE CAST(column1 AS FLOAT) > 100;"
        },
        {
            "question": "Find all records containing 'test' in column2",
            "sql": "SELECT * FROM data WHERE column2 LIKE '%test%';"
        }
    ]
}

# Cache for database connections and table info
db_cache = {}
# Memory for conversation history
conversation_memory = {}

class Query(BaseModel):
    question: str
    db_name: str
    session_id: Optional[str] = None

def get_table_info(db_path: str) -> str:
    """Get table information with sample rows."""
    if db_path in db_cache:
        return db_cache[db_path]
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='data';")
    schema = cursor.fetchone()[0]
    
    # Get sample rows
    cursor.execute("SELECT * FROM data LIMIT 3;")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    # Format table info
    table_info = f"Schema:\n{schema}\n\nSample Data:\n"
    for row in rows:
        table_info += "\t".join(str(val) for val in row) + "\n"
    
    conn.close()
    
    # Cache the result
    db_cache[db_path] = table_info
    return table_info

def select_relevant_examples(question: str) -> List[Dict]:
    """Dynamically select relevant few-shot examples based on the question."""
    examples = []
    
    # Add basic examples for all queries
    examples.extend(FEW_SHOT_EXAMPLES["basic"])
    
    # Add aggregation examples if the question contains aggregation-related words
    agg_keywords = ["average", "sum", "count", "total", "mean", "aggregate"]
    if any(keyword in question.lower() for keyword in agg_keywords):
        examples.extend(FEW_SHOT_EXAMPLES["aggregation"])
    
    # Add filtering examples if the question contains filtering-related words
    filter_keywords = ["where", "filter", "greater than", "less than", "contains", "like"]
    if any(keyword in question.lower() for keyword in filter_keywords):
        examples.extend(FEW_SHOT_EXAMPLES["filtering"])
    
    return examples

def create_optimized_prompt(table_info: str, examples: List[Dict]) -> ChatPromptTemplate:
    """Create an optimized prompt template with dynamic few-shot examples."""
    examples_str = "\n".join([
        f"Question: {ex['question']}\nSQL: {ex['sql']}"
        for ex in examples
    ])
    
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Given a question about a database, generate a SQL query to answer it.
        Use the following table information and examples as reference.
        
        Table Information:
        {table_info}
        
        Examples:
        {examples}
        
        Generate a SQL query that answers the user's question. Only output the SQL query, nothing else."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

@app.post("/upload-database")
async def upload_database(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Convert Excel/CSV to SQLite
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create SQLite database
        db_name = f"uploads/{os.path.splitext(file.filename)[0]}.db"
        conn = sqlite3.connect(db_name)
        df.to_sql('data', conn, if_exists='replace', index=False)
        conn.close()
        
        # Pre-cache table info
        get_table_info(db_name)
        
        return {"message": "Database created successfully", "db_name": os.path.basename(db_name)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_database(query: Query):
    try:
        db_path = f"uploads/{query.db_name}"
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Database not found")
        
        # Initialize or get session memory
        session_id = query.session_id or str(uuid.uuid4())
        if session_id not in conversation_memory:
            conversation_memory[session_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        
        # Get table info
        table_info = get_table_info(db_path)
        
        # Select relevant examples
        examples = select_relevant_examples(query.question)
        
        # Create optimized prompt
        prompt = create_optimized_prompt(table_info, examples)
        
        # Create SQL query chain with optimized prompt
        chain = create_sql_query_chain(
            llm,
            SQLDatabase.from_uri(f"sqlite:///{db_path}"),
            prompt=prompt
        )
        
        # Generate SQL query with conversation history
        sql_query = chain.invoke({
            "question": query.question,
            "table_info": table_info,
            "examples": examples,
            "history": conversation_memory[session_id].load_memory_variables({})["history"]
        })
        
        # Execute query
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        result = db.run(sql_query)
        
        # Update conversation memory
        conversation_memory[session_id].save_context(
            {"input": query.question},
            {"output": f"SQL Query: {sql_query}\nResult: {result}"}
        )
        
        return {
            "sql_query": sql_query,
            "result": result,
            "session_id": session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 