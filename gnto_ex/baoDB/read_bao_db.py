import sqlite3
import os
import sys

def find_db():
    # Possible locations for bao.db
    locations = [
        'bao.db',
        '../bao_server/bao.db',
        '../bao.db'
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            return loc
    return None

def inspect_db(db_path):
    print(f"Connecting to database at: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            return

        print(f"Found {len(tables)} tables.")
        
        for table in tables:
            table_name = table[0]
            print(f"\n{'='*40}")
            print(f"Table: {table_name}")
            print(f"{'='*40}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            print(f"Columns: {col_names}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Total rows: {count}")
            
            # Get sample data
            if count > 0:
                print("-" * 20)
                print("First 5 rows:")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cursor.fetchall()
                for row in rows:
                    print(row)
            else:
                print("(Empty table)")
                
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    db_path = find_db()
    
    if db_path:
        inspect_db(db_path)
    else:
        print("Could not find 'bao.db' in current directory or '../bao_server/'")
        print("Please place 'bao.db' in this directory or update the script paths.")

