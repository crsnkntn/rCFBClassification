import sqlite3

class SQLiteDataLoader:
    def __init__(self, db_file):
        self.db_file = db_file
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.cursor = self.connection.cursor()
            print("Connected to SQLite database")
        except sqlite3.Error as e:
            print("Error connecting to SQLite database:", e)

    def close(self):
        if self.connection:
            self.connection.close()
            print("Connection to SQLite database closed")

    def load_data(self, teamList, nSamples, filters):
        #
        try:
            self.cursor.execute(f"SELECT * FROM {table_name}")
            data = self.cursor.fetchall()
            return trainData, testData, validationData, trainLabels, testLabels, validationLabels
        except sqlite3.Error as e:
            print("Error loading data from SQLite database:", e)
            return None
        

        
