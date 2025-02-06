import sqlite3
import json
import numpy as np

def get_db_connection():
    """Get SQLite connection with optimized settings"""
    conn = sqlite3.connect('gate_assignments.db', isolation_level=None)  # Autocommit mode
    c = conn.cursor()
    # Disable journal file creation and synchronous writes
    c.execute('PRAGMA journal_mode=OFF')
    c.execute('PRAGMA synchronous=OFF')
    return conn, c

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    # Check if it's a numpy number
    if isinstance(obj, np.number):
        return obj.item()
    # Check if it's a numpy array
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Return the original object if it's already serializable
    return obj

def init_db():
    conn, c = get_db_connection()
    c.execute('''CREATE TABLE IF NOT EXISTS gate_assignments
                 (flight_key TEXT PRIMARY KEY, gate_info TEXT)''')
    conn.close()

def save_gate_assignment(flight_key, gate_info):
    conn, c = get_db_connection()
    
    # Convert numpy types to Python native types
    serializable_gate_info = {
        k: convert_to_serializable(v) 
        for k, v in gate_info.items()
    }
    
    # Convert flight_key to string if it's a numpy type
    serializable_key = convert_to_serializable(flight_key)
    
    c.execute('REPLACE INTO gate_assignments VALUES (?, ?)',
              (str(serializable_key), json.dumps(serializable_gate_info)))
    conn.close()

def get_gate_assignments():
    conn, c = get_db_connection()
    rows = c.execute('SELECT flight_key, gate_info FROM gate_assignments').fetchall()
    conn.close()
    return {row[0]: json.loads(row[1]) for row in rows}

def clear_db():
    """Clear all gate assignments from the database"""
    conn, c = get_db_connection()
    c.execute('DELETE FROM gate_assignments')
    conn.close() 