# MySQL Query Generator

## Overview

This project was developed iteratively, improving functionality at each stage:

1. **Initial Prototype (`test_app.py`)**

   - Started as a simple script to interact with the MySQL database.
   - Verified database connectivity and retrieved sample data.

2. **Expanding to Full Schema Retrieval (`mysql_schema_getter.py`)**

   - To gain a complete understanding of the MySQL database structure, I built a script to fetch the **full schema**.
   - Exported the schema to a CSV file for easier inspection and later use.

3. **Building a RAG-Powered Gradio App (`mysql_gradio_app.py`)**

   - Developed a **Gradio UI** to integrate **Retrieval-Augmented Generation (RAG)** using Pinecone for vector similarity search.
   - Users could view schema data and query for relevant entries via **LLM-powered search**.

4. **Refining the UI with Streamlit (`mysql_streamlit_app.py`)**

   - Discovered that **RAG was not useful** for this specific use case, so I **removed it**.
   - Replaced it with **checkbox-based schema filtering** using **Streamlit**, making the application **simpler and more effective**.
   - This version was more **successful**, offering an intuitive way to navigate the database schema.

5. **Future Plans: SSMS Integration with C# & VS**
   - To further enhance usability, I aim to **embed this functionality directly into SQL Server Management Studio (SSMS)**.
   - This will involve developing an **SSMS extension using C# and Visual Studio**.

---

## **Tech Stack**

- **Python** for scripting and backend logic.
- **pyODBC** for MySQL connectivity.
- **Gradio** for UI (earlier version, later removed).
- **Streamlit** for the final UI.
- **C# & Visual Studio** (planned) for SSMS integration.
- **Pinecone** (previously used for vector storage, later removed).
- **Groq** for large language model (LLM) API calls (previously used in RAG).
- **python-dotenv** to manage environment variables securely.

---

## **Setup Instructions**

1. **Clone or download the project.**
2. **Create a `.env` file** (not included in version control) with database credentials:

   ````ini
   MYSQL_SERVER=your_mysql_host
   MYSQL_USER=your_username
   MYSQL_PASSWORD=your_password
   MYSQL_DB=your_database

   ```bash
   bash
   CopyEdit
   MSSQL_SERVER=10.92.1.13
   MSSQL_USER=your_username
   MSSQL_PASSWORD=your_password
   MSSQL_DB=stuBook

   PINECONE_API_KEY=pcsk_...
   PINECONE_INDEX=mssql-schema
   PINECONE_NAMESPACE=ns1

   GROQ_API_KEY=gsk_...
   GROQ_MODEL=llama-3.3-70b-versatile

   ````

3. **Install dependencies**:

   ```bash
   bash
   CopyEdit
   pip install pyodbc pinecone-client gradio python-dotenv groq

   ```

4. **Run the script**:This connects to MSSQL, exports schema to CSV, upserts vectors to Pinecone, and launches Gradio on http://localhost:7860.

   ```bash
   bash
   CopyEdit
   python app.py

   ```

## Usage

- **Show Schema Table**: Click the button in Gradio to display MSSQL table/column data.
- **Query Pinecone**: Provide a 2D vector (e.g., `0.2, -0.5`) to see which rows are most similar, filtering by `genre = stuBook`.
- **Call Groq**: Enter any text prompt to test the LLM response.
