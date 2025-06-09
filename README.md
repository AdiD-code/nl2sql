# NL2SQL - Natural Language to SQL Converter

A powerful tool that converts natural language questions into SQL queries using LangChain and Google's Gemini model. Connect to your existing databases and start querying them in natural language!

## Features

- Natural language to SQL conversion
- Support for multiple database types:
  - MySQL
  - PostgreSQL
  - SQLite
  - Oracle
  - Microsoft SQL Server
- Conversation history for context-aware queries
- Beautiful Gradio interface
- Built-in query validation and safety checks
- Performance monitoring

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nl2sql.git
cd nl2sql
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Start the application:
```bash
python main.py
```

2. Open your browser and go to `http://localhost:7860`

3. Connect to your database:
   - Enter your database connection string in the format:
     - MySQL: `mysql+pymysql://username:password@host:port/database`
     - PostgreSQL: `postgresql://username:password@host:port/database`
     - SQLite: `sqlite:///path/to/your/database.db`
   - Or upload a SQLite database file directly

4. Start asking questions in natural language:
   - "Show me the top 10 customers by sales"
   - "What's the average order value this month?"
   - "List all products with stock below 100"
   - "Find customers who haven't ordered in the last 30 days"

## Example Queries

The system understands complex queries like:
- Aggregations: "What's the total revenue by product category?"
- Joins: "Show me all orders with customer details"
- Filters: "List all products with price above $100"
- Time-based: "What were the sales trends last quarter?"
- Comparisons: "Which products have above-average sales?"

## Project Structure

- `main.py`: Main application file
- `requirements.txt`: Python dependencies

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 