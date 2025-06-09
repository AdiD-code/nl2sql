import os
import requests
import zipfile
import subprocess
import mysql.connector
from pathlib import Path
import logging
import getpass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sakila():
    """Download Sakila database files"""
    logger.info("Downloading Sakila database...")
    url = "https://downloads.mysql.com/docs/sakila-db.zip"
    zip_path = "sakila-db.zip"
    
    # Download the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(chunk_size=4096):
            file.write(data)
    
    logger.info("Download completed")
    
    # Extract the zip file
    logger.info("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("sakila-db")
    
    # Clean up zip file
    os.remove(zip_path)
    logger.info("Extraction completed")

def setup_mysql_database():
    """Set up MySQL database and import Sakila data"""
    try:
        # Use root user with password
        host = "localhost"
        user = "root"
        password = input("Enter your MySQL root password: ")  # Will prompt for password
        
        # Connect to MySQL
        logger.info("Connecting to MySQL...")
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        logger.info("Creating database...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS sakila")
        cursor.execute("USE sakila")
        
        # Import schema
        logger.info("Importing schema...")
        schema_path = os.path.join("sakila-db", "sakila-db", "sakila-schema.sql")
        with open(schema_path, 'r', encoding='utf-8') as file:
            # Read the entire file
            schema_sql = file.read()
            
            # Split into individual statements
            statements = []
            current_statement = ""
            delimiter = ";"
            
            for line in schema_sql.split('\n'):
                # Skip comments and empty lines
                if line.strip().startswith('--') or not line.strip():
                    continue
                    
                # Handle DELIMITER statements
                if line.strip().upper().startswith('DELIMITER'):
                    delimiter = line.strip().split()[1]
                    continue
                    
                # Add line to current statement
                current_statement += line + '\n'
                
                # If we see the delimiter, add the statement to our list
                if line.strip().endswith(delimiter):
                    statements.append(current_statement.strip())
                    current_statement = ""
            
            # Execute each statement
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except mysql.connector.Error as err:
                        logger.warning(f"Warning executing statement: {err}")
                        continue
        
        # Import data
        logger.info("Importing data...")
        data_path = os.path.join("sakila-db", "sakila-db", "sakila-data.sql")
        with open(data_path, 'r', encoding='utf-8') as file:
            # Read the entire file
            data_sql = file.read()
            
            # Split into individual statements
            statements = []
            current_statement = ""
            
            for line in data_sql.split('\n'):
                # Skip comments and empty lines
                if line.strip().startswith('--') or not line.strip():
                    continue
                    
                # Add line to current statement
                current_statement += line + '\n'
                
                # If we see the delimiter, add the statement to our list
                if line.strip().endswith(';'):
                    statements.append(current_statement.strip())
                    current_statement = ""
            
            # Execute each statement
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except mysql.connector.Error as err:
                        logger.warning(f"Warning executing statement: {err}")
                        continue
        
        conn.commit()
        
        # Verify database setup
        logger.info("Verifying database setup...")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        if tables:
            logger.info(f"Successfully created {len(tables)} tables:")
            for table in tables:
                logger.info(f"- {table[0]}")
        else:
            logger.warning("No tables were created!")
            
        logger.info("Database setup completed successfully!")
        
        # Print connection string
        connection_string = f"mysql+pymysql://{user}:{password}@{host}/sakila"
        logger.info(f"Connection string: {connection_string}")
        
        # Save connection string to a file
        with open("mysql_connection.txt", "w") as f:
            f.write(connection_string)
        logger.info("Connection string saved to mysql_connection.txt")
        
    except mysql.connector.Error as err:
        logger.error(f"Error: {err}")
        if err.errno == 1045:  # Access denied error
            logger.error("Please check your MySQL username and password.")
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main function to set up Sakila database"""
    try:
        # Download and set up database
        download_sakila()
        setup_mysql_database()
        
        logger.info("Setup completed! You can now use the connection string in the NL2SQL application.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 