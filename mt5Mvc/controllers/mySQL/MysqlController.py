import mysql.connector
from mysql.connector import pooling
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

class MysqlController:
    """
    A controller class for handling MySQL database operations.
    This class provides methods for connecting to a MySQL database and executing queries.
    """
    
    def __init__(self, host: str = 'localhost', 
                 user: str = 'root', 
                 password: str = '137946285', 
                 database: str = 'forex', 
                 port: int = 3306,
                 pool_size: int = 5,
                 pool_name: str = 'mysql_pool'):
        """
        Initialize the MySQL controller with connection parameters.
        
        Args:
            host (str): MySQL server host
            user (str): MySQL user
            password (str): MySQL password
            database (str): MySQL database name
            port (int): MySQL server port
            pool_size (int): Connection pool size
            pool_name (str): Name for the connection pool
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.pool_size = pool_size
        self.pool_name = pool_name
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize the connection pool
        self._initialize_pool()
        
    def _initialize_pool(self) -> None:
        """Initialize the MySQL connection pool."""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            self.logger.info(f"MySQL connection pool initialized with {self.pool_size} connections")
        except mysql.connector.Error as err:
            self.logger.error(f"Failed to initialize connection pool: {err}")
            raise
    
    def get_connection(self) -> mysql.connector.MySQLConnection:
        """
        Get a connection from the pool.
        
        Returns:
            mysql.connector.MySQLConnection: A MySQL connection object
        """
        try:
            return self.connection_pool.get_connection()
        except mysql.connector.Error as err:
            self.logger.error(f"Failed to get connection from pool: {err}")
            raise
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return the results.
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Parameters for the query
            
        Returns:
            List[Dict[str, Any]]: Query results as a list of dictionaries
        """
        connection = None
        cursor = None
        results = []
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            results = cursor.fetchall()
            return results
            
        except mysql.connector.Error as err:
            self.logger.error(f"Error executing query: {err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def execute_non_query(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE) and return affected rows.
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Parameters for the query
            
        Returns:
            int: Number of affected rows
        """
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            connection.commit()
            return cursor.rowcount
            
        except mysql.connector.Error as err:
            if connection:
                connection.rollback()
            self.logger.error(f"Error executing non-query: {err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a batch operation with multiple parameter sets.
        
        Args:
            query (str): SQL query to execute
            params_list (List[tuple]): List of parameter tuples
            
        Returns:
            int: Total number of affected rows
        """
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.executemany(query, params_list)
            connection.commit()
            return cursor.rowcount
            
        except mysql.connector.Error as err:
            if connection:
                connection.rollback()
            self.logger.error(f"Error executing batch operation: {err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def execute_transaction(self, queries: List[Tuple[str, Optional[tuple]]]) -> bool:
        """
        Execute multiple queries as a single transaction.
        
        Args:
            queries (List[Tuple[str, Optional[tuple]]]): List of (query, params) tuples
            
        Returns:
            bool: True if transaction completed successfully
        """
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Start transaction
            connection.start_transaction()
            
            for query, params in queries:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
            
            # Commit transaction
            connection.commit()
            return True
            
        except mysql.connector.Error as err:
            if connection:
                connection.rollback()
            self.logger.error(f"Transaction failed: {err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def call_procedure(self, proc_name: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Call a stored procedure and return results.
        
        Args:
            proc_name (str): Name of the stored procedure
            params (tuple, optional): Parameters for the procedure
            
        Returns:
            List[Dict[str, Any]]: Results from the procedure
        """
        connection = None
        cursor = None
        results = []
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            if params:
                cursor.callproc(proc_name, params)
            else:
                cursor.callproc(proc_name)
            
            # Get results from all result sets
            for result in cursor.stored_results():
                results.extend(result.fetchall())
                
            return results
            
        except mysql.connector.Error as err:
            self.logger.error(f"Error calling procedure {proc_name}: {err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def test_connection(self) -> bool:
        """
        Test if the database connection is working.
        
        Returns:
            bool: True if connection is successful
        """
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
        except mysql.connector.Error as err:
            self.logger.error(f"Connection test failed: {err}")
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()