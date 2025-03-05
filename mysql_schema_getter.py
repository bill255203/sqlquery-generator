import mysql.connector
import csv

def get_mysql_schema_all(host, port, user, password):
    """
    Connects to MySQL and retrieves schema information for all databases.
    Returns a list of dicts for each column.
    """
    conn = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    cursor = conn.cursor()

    # Query INFORMATION_SCHEMA for columns in all user-created databases (excluding system DBs)
    query = """
    SELECT
        C.TABLE_SCHEMA,
        C.TABLE_NAME,
        C.COLUMN_NAME,
        C.DATA_TYPE,
        CASE WHEN C.COLUMN_KEY = 'PRI' THEN 'PK' ELSE '' END AS PrimaryKey
    FROM INFORMATION_SCHEMA.COLUMNS C
    WHERE C.TABLE_SCHEMA NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys')
    ORDER BY C.TABLE_SCHEMA, C.TABLE_NAME, C.ORDINAL_POSITION;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    schema_info = []
    for row in rows:
        schema_info.append({
            "DatabaseName": row[0],
            "TableName": row[1],
            "ColumnName": row[2],
            "DataType": row[3],
            "PrimaryKey": row[4],
            "ForeignKey": ""  # Foreign key detection requires additional queries
        })

    cursor.close()
    conn.close()
    return schema_info

def write_schema_to_csv(schema_data, filename="mysql_all_schemas.csv"):
    """
    Saves the schema data to CSV for quick inspection.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["DatabaseName", "TableName", "ColumnName", "DataType", "PrimaryKey", "ForeignKey"])
        for row in schema_data:
            writer.writerow([
                row["DatabaseName"],
                row["TableName"],
                row["ColumnName"],
                row["DataType"],
                row["PrimaryKey"],
                row["ForeignKey"]
            ])
    print(f"Schema data saved to {filename}.")

if __name__ == "__main__":
    # Modify credentials as needed
    HOST = "localhost"
    PORT = 3306
    USER = "root"       # Your MySQL username
    PASSWORD = "password"  # Your MySQL password

    schema_rows = get_mysql_schema_all(HOST, PORT, USER, PASSWORD)
    print(f"Fetched {len(schema_rows)} columns from all databases.")

    # Print to console for a quick look
    for row in schema_rows[:10]:  # Limit to 10 rows to avoid too much output
        print(row)

    # Optionally save to CSV
    write_schema_to_csv(schema_rows, filename="mysql_all_schemas.csv")
