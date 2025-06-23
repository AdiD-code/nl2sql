# NL2SQL: Natural Language to SQL Converter

This project lets users upload their own SQL databases and query them using natural language, powered by LLMs (Large Language Models) like Google Gemini.

## Features
- **Upload any database** (SQLite, MySQL, PostgreSQL, etc.) and query it with natural language.
- **Automatic schema analysis** and dynamic table/column selection for each question.
- **Mini-schema optimization**: Only relevant tables/columns are sent to the LLM for each query, making it fast and scalable for large databases.
- **Persistent conversation memory** (with SQLite) for context-aware, multi-turn chat.
- **Summarization** for long conversations.
- **Performance monitoring** and timing logs for each major step (table selection, LLM call, SQL execution).
- **Robust fallback**: If semantic table selection fails, falls back to keyword-based selection.
- **Configurable LLM model**: Easily change the Gemini model in `main.py` via `config.MODEL_NAME`.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your LLM provider**
   - **Google Gemini (default):**
     - Get a [Google Gemini API key](https://ai.google.dev/)
     - Set it in your environment or `.env` file:
       ```
       GOOGLE_API_KEY=your-key-here
       ```
     - The default model is `gemini-1.5-pro-002`. You can change this in `main.py`:
       ```python
       config.MODEL_NAME = "gemini-1.5-pro-002"  # or another supported model
       ```
   - **Other providers:**
     - (Planned) You can add support for OpenAI, local models, or user-supplied keys in the future.

3. **Run the app**
   ```bash
   python main.py
   ```
   - The Gradio UI will be available at http://localhost:7860

4. **Upload or connect to your database**
   - Use the UI to upload a SQLite file or enter a connection string for MySQL/PostgreSQL/etc.

5. **Ask questions!**
   - Try queries like "How many movies has Tom Hanks been in? Name them."

## Current Limitations & Notes

- **Gemini API Quota:**
  - The free tier has strict per-minute and per-day limits. If you see 429 errors, you have exceeded your quota. Wait for reset, use a different key, or upgrade your plan.
  - For deployment, consider letting users supply their own API keys or supporting multiple providers.

- **Semantic Table Selection:**
  - If semantic selection fails (e.g., due to vectorstore/embedding issues), the system falls back to keyword-based selection and logs the error.

- **LLM Model Configuration:**
  - Change the model by editing `config.MODEL_NAME` in `main.py`.

- **Performance:**
  - Timing logs are printed for table selection, mini-schema construction, LLM call, and SQL execution to help you profile and optimize.

## Next Steps / TODO
- Add support for user-supplied API keys in the UI.
- Add multi-provider fallback (Gemini, OpenAI, local models).
- Integrate dynamic few-shot example selection for each question.
- Implement caching for table selection and LLM outputs.
- Add more robust error handling and user feedback for quota issues.

---

**Contributions and feedback are welcome!** 