import os
import mysql.connector
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import groq

##############################
# LOAD ENVIRONMENT VARIABLES
##############################
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

##############################
# 1. FETCH DATABASE LIST
##############################

def get_databases():
    """Fetches all databases from the MySQL instance."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES;")
        databases = [db[0] for db in cursor.fetchall()]
        cursor.close()
        conn.close()
        return databases
    except Exception as e:
        st.error(f"Error fetching databases: {e}")
        return []

##############################
# 2. FETCH SCHEMA FROM MYSQL
##############################

def get_mysql_schema(database_name):
    """Fetches schema details for all tables in a given database."""
    schema_info = []
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=database_name
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
        
        cursor.execute(query, (database_name,))
        rows = cursor.fetchall()
        
        for row in rows:
            schema_info.append({
                "TableName": row[0],
                "ColumnName": row[1],
                "DataType": row[2],
                "PrimaryKey": row[3],
                "ForeignKey": ""  # FK not fetched yet
            })
        
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error retrieving schema for {database_name}: {e}")

    return schema_info

##############################
# 3. FETCH TABLE SCHEMA
##############################

def get_table_schema(schema_data, table_name):
    """Filters schema for the selected table."""
    table_schema = [row for row in schema_data if row["TableName"] == table_name]
    if not table_schema:
        return "Schema not found."

    schema_str = "\n".join(f"- {row['ColumnName']} ({row['DataType']})" for row in table_schema)
    return schema_str

##############################
# 4. QUERY LLM WITH SELECTED TABLES
##############################

def call_groq_llm(user_request, selected_tables, schema_data):
    """Uses LLM to generate a SQL query based on user-selected tables."""
    if not GROQ_API_KEY:
        return "Error: No GROQ_API_KEY found.", []

    schema_text = "\n\n".join(
        f"### Table: {table}\n{get_table_schema(schema_data, table)}" for table in selected_tables
    )

    prompt = f"""
    You are a SQL query generator assistant. Your job is to help non-technical users write SQL queries.

    Given the following request, generate a properly formatted SQL query:

    User request: "{user_request}"

    Below are the selected tables and their full schema:

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

    client = groq.Client(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_request}],
        max_completion_tokens=512,
        temperature=0.7
    )

    return response.choices[0].message.content, selected_tables

##############################
# 5. STREAMLIT UI
##############################

st.title("📊 MySQL Schema Explorer & SQL Generator")

# Sidebar: List databases
st.sidebar.header("📂 Select Database")
databases = get_databases()

# Default database selection
selected_db = st.sidebar.selectbox("Choose a database:", databases, index=databases.index("test_schema_db") if "test_schema_db" in databases else 0)

# Fetch schema for selected database
if selected_db:
    if f"schema_data_{selected_db}" not in st.session_state:
        st.session_state[f"schema_data_{selected_db}"] = get_mysql_schema(selected_db)

    schema_data = st.session_state[f"schema_data_{selected_db}"]
    table_names = sorted(set(row["TableName"] for row in schema_data))

    # Sidebar: Checkboxes for tables
    st.sidebar.header(f"📑 Select Tables in {selected_db}")
    selected_tables = []
    for table in table_names:
        if st.sidebar.checkbox(table, key=f"{selected_db}_{table}"):
            selected_tables.append(table)

    # Display selected tables' schema dynamically
    if selected_tables:
        st.subheader("🔍 Selected Tables Schema")
        selected_schema = pd.DataFrame([row for row in schema_data if row["TableName"] in selected_tables])
        st.dataframe(selected_schema)

# User query input
st.subheader("📝 Generate SQL Query")
user_input = st.text_area("Describe your query in plain English:", "")

# Button to generate SQL query
if st.button("🔎 Generate SQL Query"):
    if not user_input:
        st.warning("Please enter a query description.")
    elif not selected_tables:
        st.warning("Please select at least one table.")
    else:
        sql_query, used_tables = call_groq_llm(user_input, selected_tables, schema_data)
        st.subheader("📝 Generated SQL Query")
        st.text_area("SQL Query:", sql_query, height=200)
        
        st.subheader("📋 Selected Tables for Query")
        st.write(", ".join(used_tables))