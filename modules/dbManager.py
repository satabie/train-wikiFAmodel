import psycopg2


class DBManager:
    # データベースへの接続を確立する
    def __init__(self, hostname, port, dbname, user, password):
        self.dbname = dbname
        self.user = user
        self.password = password

        self.conn = psycopg2.connect(
            host=hostname,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )

    def fetch_tablenames(self):
        cur = self.conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        table_names = cur.fetchall()
        cur.close()
        return table_names

    def select(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return rows

    # INSERT文、UPDATE文、DELETE文などの変更系のクエリを実行する
    def execute(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        self.conn.commit()
        cur.close()

    # データベースへの接続をクローズする
    def close(self):
        self.conn.close()
