import os
import mysql.connector
import csv
import random
import gradio as gr
from pinecone import Pinecone
from dotenv import load_dotenv
import groq
import numpy as np
import pandas as pd

##############################
# LOAD ENVIRONMENT VARIABLES
##############################
load_dotenv()  # Loads variables from .env if present

# MySQL settings
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_DB = os.getenv("MYSQL_DB", "test_schema_db")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "mssql-schema")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CSV_FILENAME = "mysql_schema.csv"

##############################
# 1. GET SCHEMA FROM MYSQL
##############################

def get_mysql_schema():
    """
    Connects to MySQL and retrieves schema information for all tables.
    Returns a list of dictionaries.
    """
    print("Connecting to MySQL database...")
    
    schema_info = []
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()

        query = """
        SELECT
            C.TABLE_NAME,
            C.COLUMN_NAME,
            C.DATA_TYPE,
            CASE WHEN C.COLUMN_KEY = 'PRI' THEN 'PK' ELSE '' END AS PrimaryKey
        FROM INFORMATION_SCHEMA.COLUMNS C
        WHERE C.TABLE_SCHEMA = %s
        ORDER BY C.TABLE_NAME, C.ORDINAL_POSITION;
        """

        cursor.execute(query, (MYSQL_DB,))
        rows = cursor.fetchall()

        for row in rows:
            schema_info.append({
                "TableName": row[0],
                "ColumnName": row[1],
                "DataType": row[2],
                "PrimaryKey": row[3],
                "ForeignKey": ""  # Not fetching FK for now
            })

        print(f"Successfully retrieved {len(schema_info)} columns from database '{MYSQL_DB}'.")

    except Exception as e:
        print(f"Error retrieving schema: {e}")
    finally:
        cursor.close()
        conn.close()

    return schema_info

##############################
# 2. WRITE SCHEMA TO CSV
##############################

def write_schema_to_csv(schema_data, csv_filename=CSV_FILENAME):
    """
    Saves the schema data to a CSV file for inspection.
    """
    print(f"Writing schema to {csv_filename}...")
    
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TableName", "ColumnName", "DataType", "PrimaryKey", "ForeignKey"])
        for row in schema_data:
            writer.writerow([row["TableName"], row["ColumnName"], row["DataType"], row["PrimaryKey"], row["ForeignKey"]])
    
    print(f"Schema successfully saved to {csv_filename}.")

##############################
# 3. UPSERT TO PINECONE
##############################

def upsert_mysql_schema(schema_data):
    """
    Upserts one record per table instead of per column in Pinecone.
    Groups all columns under each table before inserting.
    """
    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # Get the correct vector dimension from Pinecone index
    index_stats = index.describe_index_stats()
    expected_dim = index_stats.get("dimension", 1024)  # Default to 1024 if missing

    # Step 1: Aggregate all columns per table
    table_dict = {}  # Stores columns grouped by table name
    for row in schema_data:
        table_name = row["TableName"]
        column_info = f"{row['ColumnName']} ({row['DataType']})"

        if table_name not in table_dict:
            table_dict[table_name] = []
        table_dict[table_name].append(column_info)

    # Step 2: Convert each table into a single record
    vectors = []
    for table_name, columns in table_dict.items():
        vec = np.random.rand(expected_dim).tolist()  # Generate a correct-size vector
        metadata = {
            "table_name": table_name,
            "columns": ", ".join(columns)  # Store all columns as metadata
        }

        vectors.append({
            "id": table_name,  # Unique ID per table
            "values": vec,
            "metadata": metadata
        })

    print(f"Upserting {len(vectors)} unique table vectors to Pinecone...")
    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
    print(f"Upserted {len(vectors)} unique tables into Pinecone index '{PINECONE_INDEX}' (namespace={PINECONE_NAMESPACE}).")

##############################
# 4. GROQ SQL GENERATION
##############################

# Load the schema CSV into a DataFrame (only once)
schema_df = pd.read_csv("mysql_schema.csv")

def get_table_schema(table_name):
    """
    Retrieves the full schema (column names & data types) for a given table from CSV.
    """
    table_schema = schema_df[schema_df["TableName"] == table_name]
    if table_schema.empty:
        return "Schema not found."

    # Format as "column_name (data_type)" for each column
    schema_str = "\n".join(
        f"- {row['ColumnName']} ({row['DataType']})" for _, row in table_schema.iterrows()
    )
    return schema_str

def call_groq_llm_with_rag(user_request):
    """
    Uses Pinecone to retrieve the 3 most relevant tables and then fetches their full schema 
    from CSV before sending the request to Groq LLM.
    """
    if not GROQ_API_KEY:
        return "Error: No GROQ_API_KEY found."

    print(f"DEBUG: Retrieving similar tables for: {user_request}")

    # Convert the user query into a 2D vector (mocked, should use proper embeddings)
    query_vector = np.random.rand(1024).tolist()  

    # Retrieve 3 most similar tables from Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    response = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_vector,
        top_k=3,  # Get the top 3 unique tables
        include_values=False,
        include_metadata=True
    )

    # Extract unique table names from Pinecone results
    relevant_tables = list({match["metadata"]["table_name"] for match in response["matches"]})
    print("DEBUG: Retrieved unique similar tables:", relevant_tables)

    # Ensure we always have 3 unique tables (fill empty slots if needed)
    while len(relevant_tables) < 3:
        relevant_tables.append("N/A")

    # Retrieve schema for each table
    schemas = {table: get_table_schema(table) for table in relevant_tables}

    # Create detailed table schema section
    schema_text = "\n\n".join(
        f"### Table: {table_name}\n{schemas[table_name]}"
        for table_name in relevant_tables if table_name != "N/A"
    )

    prompt = f"""
    You are a SQL query generator assistant. Your job is to help non-technical users write SQL queries.

    Given the following request, generate a properly formatted SQL query:

    User request: "{user_request}"

    Below are the 3 most relevant tables and their full schema:

    {schema_text}

    Follow these rules:
    - Use only valid SQL syntax for MySQL.
    - Ensure the query is efficient and well-structured.
    - Provide a brief natural language explanation before the SQL query.
    - **Do NOT use markdown backticks (` ``` `) in the response.**
    
    Output format:
    1. **Explanation:** [Short natural language description]
    2. **SQL Query:** [Formatted SQL query]
    """

    print("DEBUG: Prompt given to Groq LLM:\n", prompt)

    client = groq.Client(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_request}],
        max_completion_tokens=512,  # Increase to allow for longer schema context
        temperature=0.7
    )

    result = response.choices[0].message.content
    print("DEBUG: Groq response received:\n", result)
    return result, relevant_tables


##############################
# 5. GRADIO UI
##############################

def format_schema_for_display(schema_data):
    """
    Convert schema data from list of dicts to Pandas DataFrame for display.
    """
    if not schema_data:
        print("DEBUG: No schema data available.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    print(f"DEBUG: Formatting {len(schema_data)} rows of schema data into DataFrame.")
    df = pd.DataFrame(schema_data)
    print("DEBUG: DataFrame created successfully:\n", df.head())  # Show first few rows
    return df

def gradio_ui(schema_data):
    print("DEBUG: Starting Gradio UI...")
    table_data = format_schema_for_display(schema_data)

    def on_show_schema():
        print("DEBUG: Show Schema button clicked.")
        
        # Ensure table_data is a Pandas DataFrame before returning
        if isinstance(table_data, pd.DataFrame):
            print(f"DEBUG: Returning DataFrame with {len(table_data)} rows.")
            return table_data  # Return DataFrame directly
        
        print("DEBUG: table_data is NOT a DataFrame. Converting to DataFrame now.")
        df = pd.DataFrame(table_data)
        print("DEBUG: Converted DataFrame:\n", df.head())  # Debugging
        return df

    def on_query_groq(user_request):
        print(f"DEBUG: User input for SQL generation: {user_request}")
        response = call_groq_llm(user_request)
        print("DEBUG: Response from Groq:", response)
        return response

    with gr.Blocks() as demo:
        gr.Markdown(f"# MySQL Schema Explorer & SQL Generator\n")

        with gr.Row():  # Use Row to prevent full-width expansion
            with gr.Column(scale=5):  # 70% width
                gr.Markdown("### 1. Show Schema")
                btn_show_schema = gr.Button("Show Schema")
                output_table = gr.Dataframe(headers=["TableName", "ColumnName", "DataType", "PrimaryKey", "ForeignKey"], interactive=False)
                btn_show_schema.click(on_show_schema, inputs=None, outputs=output_table)

            with gr.Column(scale=5):  # 30% width
                gr.Markdown("### 2. Generate SQL Query (Groq)")
                user_input = gr.Textbox(label="Describe what you want in plain English")
                query_btn = gr.Button("Generate SQL")
                query_output = gr.Textbox(label="Generated SQL Query", lines=6)
                
                # New section for the 3 similar tables
                gr.Markdown("### 3. Similar Tables Used for RAG")
                similar_tables_output = gr.Textbox(label="Similar Tables", lines=3, interactive=False)

                # Modify query button to update SQL and similar tables
                query_btn.click(call_groq_llm_with_rag, inputs=user_input, outputs=[query_output, similar_tables_output])


    print("DEBUG: Launching Gradio UI...")
    demo.launch(server_name="127.0.0.1", server_port=7860)

##############################
# MAIN
##############################

if __name__ == "__main__":
    schema_data = get_mysql_schema()
    write_schema_to_csv(schema_data)
    upsert_mysql_schema(schema_data)
    gradio_ui(schema_data)
