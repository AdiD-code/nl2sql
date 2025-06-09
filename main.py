import os
import re
import json
import sqlite3
import pandas as pd
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache
import time
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Debug: Check if API key is loaded
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
    logger.info(f"Google API Key loaded: Yes (starts with {masked_key[:4]})")
else:
    logger.warning("Google API Key not found in environment variables")
    logger.info("Current working directory: " + os.getcwd())
    logger.info("Checking for .env file: " + str(os.path.exists('.env')))

# Core LangChain imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# Vector store and embeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Pydantic for structured output
from pydantic import BaseModel, Field

# Configuration
@dataclass
class Config:
    """Configuration class for the NL2SQL system"""
    MAX_TABLES_PER_QUERY: int = 10
    MAX_EXAMPLES_FOR_SELECTION: int = 5
    CACHE_EXPIRY_HOURS: int = 24
    MAX_QUERY_LENGTH: int = 5000
    MAX_RESULT_ROWS: int = 100
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Changed to HuggingFace model
    LLM_TEMPERATURE: float = 0.1
    MAX_CONVERSATION_HISTORY: int = 10

config = Config()

# Initialize embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="retrieval_document"
    )
except Exception as e:
    logger.warning(f"Failed to initialize Google embeddings: {e}")
    # Fallback to HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("Using HuggingFace embeddings as fallback")

class DatabaseManager:
    """Enhanced database manager with support for multiple database types"""
    
    def __init__(self):
        self.db = None
        self.db_type = None
        self.table_info_cache = {}
        self.connection_string = None
        
    def connect_database(self, db_file=None, connection_string=None, db_type="sqlite"):
        """Connect to database with enhanced error handling and type detection"""
        try:
            if db_file and db_type == "sqlite":
                # Handle SQLite file upload
                self.connection_string = f"sqlite:///{db_file}"
                self.db = SQLDatabase.from_uri(self.connection_string)
                self.db_type = "sqlite"
                
            elif connection_string:
                # Handle other database types via connection string
                self.connection_string = connection_string
                self.db = SQLDatabase.from_uri(connection_string)
                self.db_type = self._detect_db_type(connection_string)
                
            else:
                raise ValueError("Either db_file or connection_string must be provided")
                
            # Validate connection
            self._validate_connection()
            self._cache_table_info()
            
            logger.info(f"Successfully connected to {self.db_type} database")
            return True, f"Connected to {self.db_type} database with {len(self.get_table_names())} tables"
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False, f"Connection failed: {str(e)}"
    
    def _detect_db_type(self, connection_string: str) -> str:
        """Detect database type from connection string"""
        if "sqlite" in connection_string.lower():
            return "sqlite"
        elif "mysql" in connection_string.lower() or "pymysql" in connection_string.lower():
            return "mysql"
        elif "postgresql" in connection_string.lower():
            return "postgresql"
        elif "oracle" in connection_string.lower():
            return "oracle"
        elif "mssql" in connection_string.lower() or "sqlserver" in connection_string.lower():
            return "mssql"
        else:
            return "unknown"
    
    def _validate_connection(self):
        """Validate database connection"""
        if not self.db:
            raise ConnectionError("Database not connected")
        
        # Test with a simple query
        tables = self.db.get_usable_table_names()
        if not tables:
            logger.warning("No tables found in database")
    
    def _cache_table_info(self):
        """Cache table information for performance"""
        if not self.db:
            return
            
        self.table_info_cache = {
            'table_names': self.db.get_usable_table_names(),
            'table_info': self.db.table_info,
            'dialect': str(self.db.dialect),
            'cached_at': datetime.now()
        }
    
    def get_table_names(self) -> List[str]:
        """Get list of table names"""
        if self.table_info_cache:
            return self.table_info_cache['table_names']
        return self.db.get_usable_table_names() if self.db else []
    
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get table information, optionally filtered by table names"""
        if not self.db:
            return ""
            
        if table_names:
            # Get specific table info
            try:
                db_subset = SQLDatabase.from_uri(
                    self.connection_string, 
                    include_tables=table_names,
                    sample_rows_in_table_info=2
                )
                return db_subset.table_info
            except Exception as e:
                logger.error(f"Error getting subset table info: {e}")
                return self.table_info_cache.get('table_info', '')
        
        return self.table_info_cache.get('table_info', '')

class QueryCleaner:
    """Enhanced SQL query cleaner with database-specific optimizations"""
    
    @staticmethod
    def clean_sql_query(text: str, db_type: str = "sqlite") -> str:
        """Enhanced SQL query cleaning with database-specific handling"""
        if not text:
            return ""
        
        # Remove code blocks and markdown
        text = re.sub(r"```(?:sql|SQL|mysql|postgresql|sqlite)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
        
        # Remove common prefixes
        prefixes = [
            r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQLite|SQL)\s*:\s*",
            r"^(?:Answer|Response|Result)\s*:\s*",
            r"^(?:Here's|Here\s+is)\s+(?:the\s+)?(?:SQL\s+)?(?:query|statement)\s*:?\s*"
        ]
        
        for prefix in prefixes:
            text = re.sub(prefix, "", text, flags=re.IGNORECASE)
        
        # Extract SQL statement
        sql_patterns = [
            r"((?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP).*?;)",
            r"((?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP).*?)(?:\n\n|\n$|$)"
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1)
                break
        
        # Database-specific cleaning
        if db_type == "mysql":
            text = re.sub(r'`([^`]*)`', r'\1', text)  # Remove MySQL backticks
        elif db_type == "postgresql":
            text = re.sub(r'"([^"]*)"', r'\1', text)  # Remove PostgreSQL quotes
        elif db_type == "sqlite":
            text = re.sub(r'\[([^\]]*)\]', r'\1', text)  # Remove SQLite brackets
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Add semicolon if missing
        if text and not text.rstrip().endswith(';'):
            text += ';'
        
        return text

class TableSelector:
    """Advanced table selector for large databases"""
    
    def __init__(self, embeddings_model: str = None):
        self.embeddings = None
        self.vectorstore = None
        self.table_descriptions = {}
        self.table_relationships = {}
        if embeddings_model:
            self._initialize_embeddings(embeddings_model)
    
    def _initialize_embeddings(self, model_name: str):
        """Initialize embeddings model"""
        try:
            # Try HuggingFace embeddings first (free)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace embeddings: {e}")
            # Fallback to Google embeddings if API key available
            if os.getenv("GOOGLE_API_KEY"):
                self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def analyze_database_schema(self, db_manager: DatabaseManager):
        """Analyze database schema to understand table relationships"""
        if not db_manager.db:
            return
        
        table_names = db_manager.get_table_names()
        
        # Generate table descriptions from schema
        for table_name in table_names:
            try:
                # Get table info
                table_info = db_manager.db.get_table_info([table_name])
                
                # Extract column information
                columns = self._extract_columns_from_info(table_info)
                
                # Generate description
                description = self._generate_table_description(table_name, columns)
                self.table_descriptions[table_name] = description
                
                # Detect relationships
                relationships = self._detect_relationships(table_name, columns, table_names)
                self.table_relationships[table_name] = relationships
                
            except Exception as e:
                logger.error(f"Error analyzing table {table_name}: {e}")
                self.table_descriptions[table_name] = f"Table: {table_name}"
    
    def _extract_columns_from_info(self, table_info: str) -> List[Dict]:
        """Extract column information from table info string"""
        columns = []
        lines = table_info.split('\n')
        
        for line in lines:
            if 'Column' in line or '|' in line:
                # Parse column information
                parts = line.split('|') if '|' in line else [line]
                for part in parts:
                    part = part.strip()
                    if part and not part.startswith('Column'):
                        columns.append({'name': part, 'type': 'unknown'})
        
        return columns
    
    def _generate_table_description(self, table_name: str, columns: List[Dict]) -> str:
        """Generate natural language description for table"""
        col_names = [col['name'] for col in columns[:10]]  # Limit to first 10 columns
        
        # Create description based on table name and columns
        description = f"Table '{table_name}' contains information about {table_name.lower().replace('_', ' ')}"
        
        if col_names:
            description += f" with columns: {', '.join(col_names)}"
        
        # Add context based on common patterns
        if any(word in table_name.lower() for word in ['customer', 'client', 'user']):
            description += ". This table likely contains customer/user information."
        elif any(word in table_name.lower() for word in ['order', 'purchase', 'transaction']):
            description += ". This table likely contains transaction/order data."
        elif any(word in table_name.lower() for word in ['product', 'item', 'inventory']):
            description += ". This table likely contains product/inventory information."
        elif any(word in table_name.lower() for word in ['employee', 'staff', 'worker']):
            description += ". This table likely contains employee information."
        
        return description
    
    def _detect_relationships(self, table_name: str, columns: List[Dict], all_tables: List[str]) -> List[str]:
        """Detect potential relationships with other tables"""
        relationships = []
        
        # Look for foreign key patterns
        for col in columns:
            col_name = col['name'].lower()
            
            # Check if column name suggests relationship
            for other_table in all_tables:
                if other_table != table_name:
                    other_table_lower = other_table.lower()
                    
                    # Common foreign key patterns
                    if (col_name.endswith('_id') and other_table_lower in col_name) or \
                       (col_name == f"{other_table_lower}_id") or \
                       (col_name.startswith(other_table_lower) and col_name.endswith('id')):
                        relationships.append(other_table)
        
        return relationships
    
    def select_relevant_tables(self, question: str, max_tables: int = None) -> List[str]:
        """Select relevant tables for a question using semantic similarity"""
        if not self.table_descriptions:
            return list(self.table_descriptions.keys())[:max_tables or config.MAX_TABLES_PER_QUERY]
        
        if not self.embeddings:
            # Fallback to keyword matching
            return self._keyword_based_selection(question, max_tables)
        
        try:
            # Use semantic similarity
            return self._semantic_selection(question, max_tables)
        except Exception as e:
            logger.error(f"Semantic selection failed: {e}")
            return self._keyword_based_selection(question, max_tables)
    
    def _semantic_selection(self, question: str, max_tables: int) -> List[str]:
        """Select tables using semantic similarity"""
        if not self.vectorstore:
            # Initialize vector store
            documents = []
            metadatas = []
            
            for table_name, description in self.table_descriptions.items():
                documents.append(description)
                metadatas.append({"table_name": table_name})
            
            # Create embeddings for documents
            embeddings = self.embeddings.embed_documents(documents)
            
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=embeddings,
                embedding=self.embeddings,
                metadatas=metadatas
            )
        
        # Search for relevant tables
        max_tables = max_tables or config.MAX_TABLES_PER_QUERY
        results = self.vectorstore.similarity_search(question, k=max_tables)
        
        selected_tables = []
        for result in results:
            table_name = result.metadata["table_name"]
            selected_tables.append(table_name)
            
            # Add related tables
            if table_name in self.table_relationships:
                for related_table in self.table_relationships[table_name]:
                    if related_table not in selected_tables and len(selected_tables) < max_tables:
                        selected_tables.append(related_table)
        
        return selected_tables[:max_tables]
    
    def _keyword_based_selection(self, question: str, max_tables: int) -> List[str]:
        """Fallback keyword-based table selection"""
        question_lower = question.lower()
        scored_tables = []
        
        for table_name, description in self.table_descriptions.items():
            score = 0
            table_words = table_name.lower().replace('_', ' ').split()
            description_words = description.lower().split()
            
            # Score based on table name matches
            for word in table_words:
                if word in question_lower:
                    score += 2
            
            # Score based on description matches
            for word in description_words:
                if word in question_lower:
                    score += 1
            
            scored_tables.append((table_name, score))
        
        # Sort by score and return top tables
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        max_tables = max_tables or config.MAX_TABLES_PER_QUERY
        
        return [table for table, score in scored_tables[:max_tables] if score > 0]

class ExampleManager:
    """Dynamic example manager with semantic selection"""
    
    def __init__(self, embeddings_model: str = None):
        self.examples = []
        self.example_selector = None
        self.embeddings_model = embeddings_model
        self._initialize_default_examples()
    
    def _initialize_default_examples(self):
        """Initialize with comprehensive default examples"""
        self.examples = [
            {
                "input": "How many records are in the customers table?",
                "query": "SELECT COUNT(*) FROM customers;"
            },
            {
                "input": "Show me the top 10 customers by credit limit",
                "query": "SELECT customerName, creditLimit FROM customers ORDER BY creditLimit DESC LIMIT 10;"
            },
            {
                "input": "What are the different product lines available?",
                "query": "SELECT DISTINCT productLine FROM products;"
            },
            {
                "input": "Find customers who have never placed an order",
                "query": "SELECT c.customerName FROM customers c LEFT JOIN orders o ON c.customerNumber = o.customerNumber WHERE o.customerNumber IS NULL;"
            },
            {
                "input": "Get the total sales amount for each product line",
                "query": "SELECT p.productLine, SUM(od.quantityOrdered * od.priceEach) as totalSales FROM products p JOIN orderdetails od ON p.productCode = od.productCode GROUP BY p.productLine ORDER BY totalSales DESC;"
            },
            {
                "input": "Show customers from USA with credit limit over 50000",
                "query": "SELECT customerName, city, creditLimit FROM customers WHERE country = 'USA' AND creditLimit > 50000;"
            },
            {
                "input": "What is the average order value?",
                "query": "SELECT AVG(total_amount) FROM (SELECT SUM(quantityOrdered * priceEach) as total_amount FROM orderdetails GROUP BY orderNumber) as order_totals;"
            },
            {
                "input": "Find the most popular product by quantity sold",
                "query": "SELECT p.productName, SUM(od.quantityOrdered) as totalQuantity FROM products p JOIN orderdetails od ON p.productCode = od.productCode GROUP BY p.productCode ORDER BY totalQuantity DESC LIMIT 1;"
            },
            {
                "input": "List employees and their managers",
                "query": "SELECT e1.firstName + ' ' + e1.lastName as Employee, e2.firstName + ' ' + e2.lastName as Manager FROM employees e1 LEFT JOIN employees e2 ON e1.reportsTo = e2.employeeNumber;"
            },
            {
                "input": "Show monthly sales trends for this year",
                "query": "SELECT MONTH(o.orderDate) as month, SUM(od.quantityOrdered * od.priceEach) as monthlySales FROM orders o JOIN orderdetails od ON o.orderNumber = od.orderNumber WHERE YEAR(o.orderDate) = YEAR(CURDATE()) GROUP BY MONTH(o.orderDate) ORDER BY month;"
            }
        ]
    
    def add_examples_from_database(self, db_manager: DatabaseManager):
        """Add database-specific examples based on schema analysis"""
        if not db_manager.db:
            return
        
        table_names = db_manager.get_table_names()
        
        # Generate examples for each table
        for table_name in table_names[:5]:  # Limit to first 5 tables
            try:
                # Add basic count example
                self.examples.append({
                    "input": f"How many records are in {table_name}?",
                    "query": f"SELECT COUNT(*) FROM {table_name};"
                })
                
                # Add selection example
                self.examples.append({
                    "input": f"Show me all data from {table_name}",
                    "query": f"SELECT * FROM {table_name} LIMIT 10;"
                })
                
            except Exception as e:
                logger.error(f"Error adding examples for table {table_name}: {e}")
    
    def initialize_selector(self, embeddings):
        """Initialize semantic similarity example selector"""
        try:
            if embeddings and len(self.examples) > 0:
                self.example_selector = SemanticSimilarityExampleSelector.from_examples(
                    self.examples,
                    embeddings,
                    FAISS,
                    k=config.MAX_EXAMPLES_FOR_SELECTION,
                    input_keys=["input"]
                )
        except Exception as e:
            logger.error(f"Error initializing example selector: {e}")
    
    def get_relevant_examples(self, question: str, k: int = None) -> List[Dict]:
        """Get relevant examples for a question"""
        k = k or config.MAX_EXAMPLES_FOR_SELECTION
        
        if self.example_selector:
            try:
                return self.example_selector.select_examples({"input": question})
            except Exception as e:
                logger.error(f"Error selecting examples: {e}")
        
        # Fallback to simple keyword matching
        return self._keyword_based_example_selection(question, k)
    
    def _keyword_based_example_selection(self, question: str, k: int) -> List[Dict]:
        """Fallback keyword-based example selection"""
        question_lower = question.lower()
        scored_examples = []
        
        for example in self.examples:
            score = 0
            example_words = example["input"].lower().split()
            
            for word in example_words:
                if word in question_lower:
                    score += 1
            
            scored_examples.append((example, score))
        
        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        return [example for example, score in scored_examples[:k] if score > 0]

class ConversationManager:
    """Enhanced conversation manager with context awareness"""
    
    def __init__(self):
        self.histories = {}  # Multiple conversation histories
        self.default_session = "default"
    
    def get_history(self, session_id: str = None) -> ChatMessageHistory:
        """Get or create conversation history for session"""
        session_id = session_id or self.default_session
        
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
        
        return self.histories[session_id]
    
    def add_interaction(self, question: str, response: str, session_id: str = None):
        """Add question-response pair to conversation history"""
        history = self.get_history(session_id)
        history.add_user_message(question)
        history.add_ai_message(response)
        
        # Limit history size
        if len(history.messages) > config.MAX_CONVERSATION_HISTORY * 2:
            # Remove oldest messages (keep pairs)
            history.messages = history.messages[-config.MAX_CONVERSATION_HISTORY * 2:]
    
    def get_context_messages(self, session_id: str = None) -> List:
        """Get context messages for the conversation"""
        history = self.get_history(session_id)
        return history.messages
    
    def clear_history(self, session_id: str = None):
        """Clear conversation history"""
        session_id = session_id or self.default_session
        if session_id in self.histories:
            del self.histories[session_id]

class NL2SQLChain:
    """Main NL2SQL chain orchestrator"""
    
    def __init__(self):
        self.llm = None
        self.db_manager = DatabaseManager()
        self.query_cleaner = QueryCleaner()
        self.table_selector = TableSelector(config.EMBEDDING_MODEL)
        self.example_manager = ExampleManager()
        self.conversation_manager = ConversationManager()
        self.query_executor = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            # Try Google Gemini first (free tier available)
            if os.getenv("GOOGLE_API_KEY"):
                self.llm = GoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=config.LLM_TEMPERATURE,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                logger.info("Initialized Google Gemini model")
            else:
                raise ValueError("GOOGLE_API_KEY not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    def setup_database(self, db_file=None, connection_string=None, db_type="sqlite"):
        """Setup database connection and analyze schema"""
        success, message = self.db_manager.connect_database(db_file, connection_string, db_type)
        
        if success:
            # Analyze database schema
            self.table_selector.analyze_database_schema(self.db_manager)
            
            # Add database-specific examples
            self.example_manager.add_examples_from_database(self.db_manager)
            
            # Initialize example selector
            if self.table_selector.embeddings:
                self.example_manager.initialize_selector(self.table_selector.embeddings)
            
            # Setup query executor
            self.query_executor = QuerySQLDataBaseTool(db=self.db_manager.db)
            
            logger.info("Database setup completed successfully")
        
        return success, message
    
    def process_question(self, question: str, session_id: str = None) -> Tuple[str, str, str]:
        """Process a natural language question and return SQL query, results, and formatted response"""
        try:
            # Validate input
            if not question or not question.strip():
                return "", "", "Please enter a valid question."
            
            # Check database connection
            if not self.db_manager.db:
                return "", "", "Please connect to a database first."
            
            logger.info(f"Processing question: {question}")
            
            # Validate database info
            if not self.db_manager.db.table_info:
                logger.error("No table information available")
                return "", "", "Database schema information not available. Please reconnect to the database."
            
            logger.info("Database schema loaded successfully")
            
            # Get conversation history
            history = self.conversation_manager.get_history(session_id)
            logger.info(f"Conversation history length: {len(history.messages)}")
            
            # Create the SQL query chain with conversation history
            try:
                query_chain = create_sql_query_chain(
                    llm=self.llm,
                    db=self.db_manager.db,
                    k=5  # Number of examples to use
                )
                logger.info("SQL query chain created successfully")
            except Exception as e:
                logger.error(f"Failed to create SQL query chain: {e}")
                return "", "", "Failed to initialize query generation. Please try again."
            
            # Generate query
            try:
                raw_query = query_chain.invoke({
                    "question": question,
                    "table_info": self.db_manager.db.table_info
                })
                logger.info(f"Generated raw query: {raw_query}")
            except Exception as e:
                logger.error(f"Failed to generate query: {e}")
                return "", "", "Failed to generate SQL query. Please try rephrasing your question."
            
            # Clean query
            cleaned_query = self.query_cleaner.clean_sql_query(raw_query, self.db_manager.db_type)
            logger.info(f"Cleaned query: {cleaned_query}")
            
            if not cleaned_query:
                return "", "", "Failed to generate a valid SQL query. Please try rephrasing your question."
            
            # Execute query
            try:
                result = self.query_executor.invoke(cleaned_query)
                logger.info("Query executed successfully")
                
                # Format response
                formatted_response = self._format_response(question, cleaned_query, result)
                
                # Add to conversation history
                history.add_user_message(question)
                history.add_ai_message(formatted_response)
                logger.info("Added interaction to conversation history")
                
                return cleaned_query, str(result), formatted_response
                
            except Exception as e:
                error_msg = f"Query execution failed: {str(e)}"
                logger.error(error_msg)
                return cleaned_query, "", error_msg
                
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return "", "", error_msg
    
    def _format_response(self, question: str, query: str, result: str) -> str:
        """Format the final response using LLM"""
        try:
            format_prompt = ChatPromptTemplate.from_template(
                """Given the user question, SQL query, and query results, provide a clear and helpful answer.

Question: {question}
SQL Query: {query}
Results: {result}

Please provide a natural language response that:
1. Directly answers the user's question
2. Highlights key findings from the data
3. Is easy to understand for non-technical users
4. Mentions any limitations or assumptions

Answer:"""
            )
            
            format_chain = format_prompt | self.llm | StrOutputParser()
            
            response = format_chain.invoke({
                "question": question,
                "query": query,
                "result": result
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return f"Query executed successfully. Results: {result}"

# Initialize the main NL2SQL system
nl2sql_system = NL2SQLChain()

def setup_database_interface(db_file, connection_string, db_type):
    """Interface function for database setup"""
    try:
        if db_file:
            success, message = nl2sql_system.setup_database(db_file=db_file.name, db_type="sqlite")
        elif connection_string:
            success, message = nl2sql_system.setup_database(connection_string=connection_string, db_type=db_type)
        else:
            return "Please provide either a database file or connection string."
        
        if success:
            tables = nl2sql_system.db_manager.get_table_names()
            table_info = f"Successfully connected! Found {len(tables)} tables: {', '.join(tables[:10])}"
            if len(tables) > 10:
                table_info += f"... and {len(tables) - 10} more"
            return table_info
        else:
            return f"Connection failed: {message}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def process_query_interface(question, session_id="default"):
    """Interface function for processing queries"""
    try:
        if not question.strip():
            return "", "", "Please enter a question."
        
        sql_query, raw_result, formatted_response = nl2sql_system.process_question(question, session_id)
        
        return sql_query, raw_result, formatted_response
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        return "", "", error_msg

def clear_conversation_interface(session_id="default"):
    """Interface function for clearing conversation history"""
    nl2sql_system.conversation_manager.clear_history(session_id)
    return "Conversation history cleared."

def get_table_info_interface():
    """Interface function to get table information"""
    try:
        if not nl2sql_system.db_manager.db:
            return "No database connected."
        
        tables = nl2sql_system.db_manager.get_table_names()
        table_descriptions = nl2sql_system.table_selector.table_descriptions
        
        info = f"Database contains {len(tables)} tables:\n\n"
        
        for table in tables[:20]:  # Show first 20 tables
            if table in table_descriptions:
                info += f"• {table}: {table_descriptions[table]}\n"
            else:
                info += f"• {table}\n"
        
        if len(tables) > 20:
            info += f"\n... and {len(tables) - 20} more tables"
            
        return info
        
    except Exception as e:
        return f"Error getting table info: {str(e)}"

# Performance monitoring and caching
class PerformanceMonitor:
    """Monitor and optimize system performance"""
    
    def __init__(self):
        self.query_cache = {}
        self.performance_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0,
            'cache_hits': 0
        }
    
    @lru_cache(maxsize=100)
    def cached_table_selection(self, question_hash: str, table_names_hash: str):
        """Cache table selection results"""
        pass
    
    def log_query_performance(self, question: str, response_time: float, success: bool):
        """Log query performance metrics"""
        self.performance_stats['total_queries'] += 1
        if success:
            self.performance_stats['successful_queries'] += 1
        
        # Update average response time
        current_avg = self.performance_stats['average_response_time']
        total_queries = self.performance_stats['total_queries']
        self.performance_stats['average_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
    
    def get_stats(self):
        """Get performance statistics"""
        return self.performance_stats

# Add performance monitoring to the system
performance_monitor = PerformanceMonitor()

# Enhanced error handling and validation
class QueryValidator:
    """Validate and sanitize SQL queries"""
    
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
        'EXEC', 'EXECUTE', 'sp_', 'xp_', 'MERGE', 'BULK'
    ]
    
    @staticmethod
    def is_safe_query(query: str) -> Tuple[bool, str]:
        """Check if query is safe for execution"""
        query_upper = query.upper()
        
        # Check for dangerous keywords
        for keyword in QueryValidator.DANGEROUS_KEYWORDS:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous keyword: {keyword}"
        
        # Check query length
        if len(query) > config.MAX_QUERY_LENGTH:
            return False, f"Query too long (max {config.MAX_QUERY_LENGTH} characters)"
        
        # Ensure it's a SELECT query (for safety)
        if not query_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, "Query is safe"
    
    @staticmethod
    def validate_question(question: str) -> Tuple[bool, str]:
        """Validate user question"""
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if len(question) > 1000:
            return False, "Question too long (max 1000 characters)"
        
        # Check for potential injection attempts
        suspicious_patterns = ['--', '/*', '*/', ';--', 'union select', 'drop table']
        question_lower = question.lower()
        
        for pattern in suspicious_patterns:
            if pattern in question_lower:
                return False, f"Question contains suspicious pattern: {pattern}"
        
        return True, "Question is valid"

# Add query validator to the system
query_validator = QueryValidator()

# Gradio Interface
def create_gradio_interface():
    """Create the Gradio interface for the NL2SQL application"""
    with gr.Blocks(title="NL2SQL - Natural Language to SQL Converter") as interface:
        gr.Markdown("# NL2SQL - Natural Language to SQL Converter")
        
        # Load MySQL connection string
        connection_string = setup_environment()
        if connection_string:
            gr.Markdown("### Connected to Sakila Database")
            gr.Markdown("The application is connected to the Sakila sample database.")
            gr.Markdown("Try queries like:")
            gr.Markdown("- 'Show me all films starring Tom Hanks'")
            gr.Markdown("- 'List all customers who rented action movies'")
            gr.Markdown("- 'What are the top 10 most rented films?'")
            gr.Markdown("- 'Show me the total sales by store'")
        
        # Database Setup Section
        with gr.Tab("🗄️ Database Setup"):
            gr.Markdown("### Connect Your Database")
            
            with gr.Row():
                with gr.Column():
                    db_file = gr.File(
                        label="Upload SQLite Database File",
                        file_types=[".db", ".sqlite", ".sqlite3"],
                        type="filepath"
                    )
                    
                with gr.Column():
                    connection_string = gr.Textbox(
                        label="Or Enter Connection String",
                        placeholder="mysql+pymysql://user:password@host:port/database",
                        lines=2
                    )
                    
                    db_type = gr.Dropdown(
                        choices=["sqlite", "mysql", "postgresql", "oracle", "mssql"],
                        label="Database Type",
                        value="sqlite"
                    )
            
            setup_btn = gr.Button("🔗 Connect Database", variant="primary")
            connection_status = gr.Textbox(label="Connection Status", interactive=False)
            
            # Table Information
            with gr.Accordion("📋 View Table Information", open=False):
                table_info_btn = gr.Button("📊 Show Table Information")
                table_info_display = gr.Textbox(
                    label="Database Schema",
                    lines=10,
                    interactive=False
                )
        
        # Query Interface Section  
        with gr.Tab("🤖 Ask Questions"):
            gr.Markdown("### Ask Questions About Your Data")
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., How many customers are from France?",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Ask Question", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Example Questions")
                    gr.Markdown("""
                    - How many records are in each table?
                    - Show me the top 10 customers by sales
                    - Which products are most popular?
                    - What's the average order value?
                    - List customers who haven't ordered recently
                    """)
            
            # Results Section
            with gr.Row():
                with gr.Column():
                    sql_output = gr.Code(
                        label="Generated SQL Query",
                        language="sql",
                        interactive=False
                    )
                    
                with gr.Column():
                    raw_results = gr.Textbox(
                        label="Raw Query Results",
                        lines=8,
                        interactive=False
                    )
            
            # Formatted Response
            formatted_response = gr.Textbox(
                label="📝 AI Response",
                lines=6,
                interactive=False
            )
        
        # Advanced Settings
        with gr.Tab("⚙️ Advanced Settings"):
            gr.Markdown("### System Configuration")
            
            with gr.Row():
                with gr.Column():
                    max_tables = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=config.MAX_TABLES_PER_QUERY,
                        step=1,
                        label="Maximum Tables per Query"
                    )
                    
                    max_results = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        value=config.MAX_RESULT_ROWS,
                        step=10,
                        label="Maximum Result Rows"
                    )
                
                with gr.Column():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.LLM_TEMPERATURE,
                        step=0.1,
                        label="LLM Temperature"
                    )
                    
                    similarity_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.DEFAULT_SIMILARITY_THRESHOLD,
                        step=0.1,
                        label="Similarity Threshold"
                    )
            
            # API Key Settings
            with gr.Accordion("🔑 API Configuration", open=False):
                google_api_key = gr.Textbox(
                    label="Google API Key (for Gemini)",
                    placeholder="Enter your Google API key here",
                    type="password"
                )
                
                save_api_key_btn = gr.Button("💾 Save API Key")
                api_status = gr.Textbox(label="API Status", interactive=False)
        
        # Performance Monitoring
        with gr.Tab("📊 Performance"):
            gr.Markdown("### System Performance Metrics")
            
            refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
            
            with gr.Row():
                total_queries_stat = gr.Number(label="Total Queries", interactive=False)
                success_rate_stat = gr.Number(label="Success Rate (%)", interactive=False)
                avg_time_stat = gr.Number(label="Avg Response Time (s)", interactive=False)
                cache_hits_stat = gr.Number(label="Cache Hits", interactive=False)
        
        # Event Handlers
        def setup_database_wrapper(db_file, conn_str, db_type):
            start_time = time.time()
            try:
                result = setup_database_interface(db_file, conn_str, db_type)
                success = not result.startswith("Connection failed") and not result.startswith("Error")
                performance_monitor.log_query_performance("database_setup", time.time() - start_time, success)
                return result
            except Exception as e:
                performance_monitor.log_query_performance("database_setup", time.time() - start_time, False)
                return f"Setup error: {str(e)}"
        
        def process_question_wrapper(question):
            start_time = time.time()
            try:
                # Validate question
                is_valid, validation_msg = query_validator.validate_question(question)
                if not is_valid:
                    return "", "", validation_msg
                
                sql_query, raw_result, formatted_response = process_query_interface(question)
                
                # Validate generated SQL
                if sql_query:
                    is_safe, safety_msg = query_validator.is_safe_query(sql_query)
                    if not is_safe:
                        return sql_query, "", f"Query blocked for safety: {safety_msg}"
                
                success = bool(formatted_response and not formatted_response.startswith("Error"))
                performance_monitor.log_query_performance(question, time.time() - start_time, success)
                
                return sql_query, raw_result, formatted_response
                
            except Exception as e:
                performance_monitor.log_query_performance(question, time.time() - start_time, False)
                return "", "", f"Processing error: {str(e)}"
        
        def save_api_key_wrapper(api_key):
            try:
                if api_key:
                    os.environ["GOOGLE_API_KEY"] = api_key
                    # Reinitialize the LLM
                    nl2sql_system._initialize_llm()
                    return "API key saved and LLM reinitialized successfully!"
                else:
                    return "Please enter a valid API key."
            except Exception as e:
                return f"Error saving API key: {str(e)}"
        
        def get_performance_stats():
            stats = performance_monitor.get_stats()
            success_rate = (stats['successful_queries'] / max(stats['total_queries'], 1)) * 100
            return (
                stats['total_queries'],
                round(success_rate, 2),
                round(stats['average_response_time'], 3),
                stats['cache_hits']
            )
        
        def update_config(max_tables, max_results, temperature, similarity_threshold):
            config.MAX_TABLES_PER_QUERY = int(max_tables)
            config.MAX_RESULT_ROWS = int(max_results)
            config.LLM_TEMPERATURE = temperature
            config.DEFAULT_SIMILARITY_THRESHOLD = similarity_threshold
            
            # Update LLM temperature if possible
            if hasattr(nl2sql_system.llm, 'temperature'):
                nl2sql_system.llm.temperature = temperature
            
            return "Configuration updated successfully!"
        
        # Wire up the interface
        setup_btn.click(
            fn=setup_database_wrapper,
            inputs=[db_file, connection_string, db_type],
            outputs=[connection_status]
        )
        
        table_info_btn.click(
            fn=get_table_info_interface,
            outputs=[table_info_display]
        )
        
        submit_btn.click(
            fn=process_question_wrapper,
            inputs=[question_input],
            outputs=[sql_output, raw_results, formatted_response]
        )
        
        clear_btn.click(
            fn=clear_conversation_interface,
            outputs=[]
        )
        
        save_api_key_btn.click(
            fn=save_api_key_wrapper,
            inputs=[google_api_key],
            outputs=[api_status]
        )
        
        refresh_stats_btn.click(
            fn=get_performance_stats,
            outputs=[total_queries_stat, success_rate_stat, avg_time_stat, cache_hits_stat]
        )
        
        # Auto-update config when sliders change
        for slider in [max_tables, max_results, temperature, similarity_threshold]:
            slider.change(
                fn=update_config,
                inputs=[max_tables, max_results, temperature, similarity_threshold],
                outputs=[]
            )
    
    return interface

# Additional utility functions for deployment
def setup_environment():
    """Set up the environment and initialize components"""
    try:
        # Load MySQL connection string if available
        mysql_conn_file = "mysql_connection.txt"
        if os.path.exists(mysql_conn_file):
            with open(mysql_conn_file, 'r') as f:
                connection_string = f.read().strip()
                logger.info(f"Loaded MySQL connection string from {mysql_conn_file}")
                return connection_string
        return None
    except Exception as e:
        logger.error(f"Error in setup_environment: {e}")
        return None

def create_sample_database():
    """Create a sample SQLite database for testing"""
    sample_db_path = "sample_database.db"
    
    if os.path.exists(sample_db_path):
        return sample_db_path
    
    conn = sqlite3.connect(sample_db_path)
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            country TEXT,
            credit_limit REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            total_amount REAL,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_quantity INTEGER
        )
    """)
    
    # Insert sample data
    customers_data = [
        (1, 'John Doe', 'john@email.com', 'USA', 5000.0),
        (2, 'Jane Smith', 'jane@email.com', 'Canada', 7500.0),
        (3, 'Pierre Dubois', 'pierre@email.com', 'France', 10000.0),
    ]
    
    orders_data = [
        (1, 1, '2024-01-15', 299.99),
        (2, 2, '2024-01-20', 149.50),
        (3, 1, '2024-02-01', 89.99),
    ]
    
    products_data = [
        (1, 'Laptop', 'Electronics', 999.99, 50),
        (2, 'Mouse', 'Electronics', 29.99, 200),
        (3, 'Desk Chair', 'Furniture', 199.99, 25),
    ]
    
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers_data)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders_data)
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created sample database: {sample_db_path}")
    return sample_db_path

# Main execution
if __name__ == "__main__":
    # Setup environment
    env_ready = setup_environment()
    
    if not env_ready:
        print("Setting up with sample database for testing...")
        sample_db = create_sample_database()
        print(f"Sample database created at: {sample_db}")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    # Launch configuration
    interface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default port for Hugging Face Spaces
        share=True,             # Create shareable link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed errors
    )

# Export key functions for programmatic use
__all__ = [
    'NL2SQLChain',
    'DatabaseManager', 
    'QueryCleaner',
    'TableSelector',
    'ExampleManager',
    'ConversationManager',
    'create_gradio_interface',
    'setup_environment',
    'create_sample_database'
]