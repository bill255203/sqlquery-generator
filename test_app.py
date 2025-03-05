import os
import pyodbc
import csv
import random
import gradio as gr
from pinecone import Pinecone
from dotenv import load_dotenv
import groq

##############################
# LOAD ENVIRONMENT VARIABLES
##############################
load_dotenv()  # Loads variables from .env if present

# MSSQL settings
MSSQL_SERVER = os.getenv("MSSQL_SERVER", "10.92.1.13")
MSSQL_USER = os.getenv("MSSQL_USER", "your_username")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "your_password")
MSSQL_DB = os.getenv("MSSQL_DB", "stuBook")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "mssql-schema")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CSV_FILENAME = "stuBook_schema.csv"

##############################
# 1. GET SCHEMA FROM MSSQL
##############################

def get_stuBook_db_schema():
    """
    Connect to the DB on MSSQL and gather table/column info with PK/FK.
    Returns a list of dictionaries.
    """
    connection_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={MSSQL_SERVER};"
        f"UID={MSSQL_USER};"
        f"PWD={MSSQL_PASSWORD};"
        f"DATABASE={MSSQL_DB};"
    )
    schema_info = []

    with pyodbc.connect(connection_str) as conn:
        cursor = conn.cursor()

        query = """
        SELECT
            t.name AS TableName,
            c.name AS ColumnName,
            ty.name AS DataType,
            CASE WHEN pkc.column_id IS NOT NULL THEN 'PK' ELSE '' END AS PrimaryKey,
            CASE WHEN fkc.parent_object_id IS NOT NULL THEN 'FK' ELSE '' END AS ForeignKey
        FROM sys.tables t
        INNER JOIN sys.columns c ON t.object_id = c.object_id
        INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id
        LEFT JOIN (
            SELECT i.object_id, ic.column_id
            FROM sys.indexes i
            INNER JOIN sys.index_columns ic
                ON i.object_id = ic.object_id
                AND i.index_id = ic.index_id
            WHERE i.is_primary_key = 1
        ) AS pkc ON t.object_id = pkc.object_id AND c.column_id = pkc.column_id
        LEFT JOIN (
            SELECT parent_object_id, parent_column_id
            FROM sys.foreign_key_columns
        ) AS fkc ON t.object_id = fkc.parent_object_id AND c.column_id = fkc.parent_column_id
        ORDER BY TableName, c.column_id;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            schema_info.append({
                "TableName": row[0],
                "ColumnName": row[1],
                "DataType": row[2],
                "PrimaryKey": row[3],
                "ForeignKey": row[4]
            })

    return schema_info

##############################
# 2. WRITE SCHEMA TO CSV
##############################

def write_schema_to_csv(schema_data, csv_filename=CSV_FILENAME):
    """
    Saves the schema data to a CSV file for inspection.
    """
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["TableName", "ColumnName", "DataType", "PrimaryKey", "ForeignKey"])
        # Rows
        for row in schema_data:
            writer.writerow([
                row["TableName"],
                row["ColumnName"],
                row["DataType"],
                row["PrimaryKey"],
                row["ForeignKey"]
            ])
    print(f"Schema data saved to {csv_filename}.")

##############################
# 3. UPSERT TO PINECONE
##############################

def upsert_stuBook_schema(schema_data):
    """
    Upserts each row into Pinecone with random 2D vectors for testing.
    We'll store DB name in 'genre', plus table/column info.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    vectors = []
    for i, row in enumerate(schema_data):
        vec = [random.uniform(-1, 1), random.uniform(-1, 1)]
        metadata = {
            "genre": MSSQL_DB,
            "table_name": row["TableName"],
            "column_name": row["ColumnName"]
        }

        vectors.append({
            "id": f"row-{i}",
            "values": vec,
            "metadata": metadata
        })

    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
    print(f"Upserted {len(vectors)} vectors into Pinecone index '{PINECONE_INDEX}'. (ns={PINECONE_NAMESPACE})")

##############################
# 4. QUERY PINECONE
##############################

def query_stuBook_pinecone(vector_str):
    """
    Query Pinecone with a user-provided 2D vector string (e.g. '0.1, 0.3').
    Filter by 'genre' == MSSQL_DB
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    try:
        floats = [float(x.strip()) for x in vector_str.split(",")]
    except:
        floats = [0.0, 0.0]

    response = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=floats,
        top_k=3,
        include_values=True,
        include_metadata=True,
        filter={"genre": {"$eq": MSSQL_DB}}
    )
    return response

##############################
# 5. GROQ EXAMPLE
##############################

def call_groq_llm(prompt):
    """
    Example: call Groq chat.completions with a short system prompt plus user prompt.
    Return the text response.
    """
    if not GROQ_API_KEY:
        return "Error: No GROQ_API_KEY found in environment."

    client = groq.Client(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=256,
        temperature=0.7
    )
    return response.choices[0].message.content

##############################
# 6. GRADIO UI
##############################

def build_schema_table(schema_data):
    table = [["TableName", "ColumnName", "DataType", "PK", "FK"]]
    for row in schema_data:
        table.append([
            row["TableName"],
            row["ColumnName"],
            row["DataType"],
            row["PrimaryKey"],
            row["ForeignKey"]
        ])
    return table

def gradio_ui(schema_data):
    table_data = build_schema_table(schema_data)

    def on_show_schema():
        return table_data

    def on_query_pinecone(vector_text):
        resp = query_stuBook_pinecone(vector_text)
        return str(resp)

    def on_call_groq_llm(user_prompt):
        result = call_groq_llm(user_prompt)
        return result

    with gr.Blocks() as demo:
        gr.Markdown(f"# MSSQL Schema & Pinecone & Groq Demo\n")
        gr.Markdown(f"**Database**: {MSSQL_DB} | **Pinecone Index**: {PINECONE_INDEX} | **Groq Model**: {GROQ_MODEL}")

        with gr.Box():
            gr.Markdown("### 1. Show Exported Schema")
            btn_show_schema = gr.Button("Show Schema Table")
            output_table = gr.Dataframe()
            btn_show_schema.click(on_show_schema, inputs=None, outputs=output_table)

        with gr.Box():
            gr.Markdown("### 2. Query Pinecone")
            vector_input = gr.Textbox(label="Enter 2D vector (e.g. '0.1,0.3')", value="0.0,0.0")
            btn_query = gr.Button("Search Pinecone")
            result_text = gr.Textbox(label="Query Result", lines=6)
            btn_query.click(on_query_pinecone, inputs=vector_input, outputs=result_text)

        with gr.Box():
            gr.Markdown("### 3. Test Groq LLM")
            groq_input = gr.Textbox(label="Enter your prompt for Groq", value="Explain the importance of AI.")
            groq_btn = gr.Button("Call Groq")
            groq_output = gr.Textbox(label="Groq LLM Response", lines=6)
            groq_btn.click(on_call_groq_llm, inputs=groq_input, outputs=groq_output)

    demo.launch(server_name="0.0.0.0", server_port=7860)

##############################
# MAIN
##############################

if __name__ == "__main__":
    # 1. Fetch schema from MSSQL
    schema_data = get_stuBook_db_schema()
    print(f"Found {len(schema_data)} columns in '{MSSQL_DB}' database.")

    # 2. Write to CSV
    write_schema_to_csv(schema_data)

    # 3. Upsert to Pinecone
    upsert_stuBook_schema(schema_data)

    # 4. Launch Gradio UI (including a Groq test)
    gradio_ui(schema_data)
