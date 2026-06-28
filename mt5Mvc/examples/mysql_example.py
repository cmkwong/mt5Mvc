#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of the MysqlController class.
"""

import sys
import os
import logging

# Add the parent directory to the path to import the controller
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.mySQL.MysqlController import MysqlController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Example function demonstrating MysqlController usage."""
    
    # Replace these with your actual database credentials
    db_config = {
        'host': 'localhost',
        'user': 'your_username',
        'password': 'your_password',
        'database': 'your_database',
        'port': 3306
    }
    
    try:
        # Initialize the controller
        mysql_controller = MysqlController(**db_config)
        
        # Test connection
        if mysql_controller.test_connection():
            print("Successfully connected to the database!")
        else:
            print("Failed to connect to the database.")
            return
        
        # Example: Execute a simple SELECT query
        results = mysql_controller.execute_query("SELECT * FROM your_table LIMIT 5")
        print(f"Query results: {results}")
        
        # Example: Insert data
        insert_query = "INSERT INTO your_table (column1, column2) VALUES (%s, %s)"
        params = ('value1', 'value2')
        affected_rows = mysql_controller.execute_non_query(insert_query, params)
        print(f"Inserted {affected_rows} row(s)")
        
        # Example: Update data
        update_query = "UPDATE your_table SET column1 = %s WHERE column2 = %s"
        params = ('new_value', 'value2')
        affected_rows = mysql_controller.execute_non_query(update_query, params)
        print(f"Updated {affected_rows} row(s)")
        
        # Example: Execute a transaction
        transaction_queries = [
            ("INSERT INTO your_table (column1, column2) VALUES (%s, %s)", ('tx_value1', 'tx_value2')),
            ("UPDATE your_table SET column1 = %s WHERE column2 = %s", ('tx_updated', 'tx_value2'))
        ]
        success = mysql_controller.execute_transaction(transaction_queries)
        print(f"Transaction completed successfully: {success}")
        
        # Example: Batch insert
        batch_query = "INSERT INTO your_table (column1, column2) VALUES (%s, %s)"
        batch_params = [
            ('batch1', 'value1'),
            ('batch2', 'value2'),
            ('batch3', 'value3')
        ]
        affected_rows = mysql_controller.execute_many(batch_query, batch_params)
        print(f"Batch insert affected {affected_rows} row(s)")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()