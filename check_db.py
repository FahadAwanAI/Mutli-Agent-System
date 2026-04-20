import sqlite3
import json

def check_db():
    try:
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        
        print("--- Events ---")
        cursor.execute("SELECT * FROM events")
        for row in cursor.fetchall():
            print(row)
            
        print("\n--- Predictions ---")
        cursor.execute("SELECT * FROM predictions")
        for row in cursor.fetchall():
            print(row)
            
        conn.close()
    except Exception as e:
        print(f"Error checking DB: {e}")

if __name__ == "__main__":
    check_db()
