import streamlit as st
import pandas as pd
import ast, re
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

# -----------------------------------
# DB & Modell
# -----------------------------------
db = SQLDatabase.from_uri("postgresql+psycopg://postgres:carsten25@localhost:5432/chinook1")
model = ChatOllama(model="llama3.1")

# -----------------------------------
# Intent-Erkennung
# -----------------------------------
def detect_intent(q: str) -> str:
    ql = q.lower()

    # Schlüsselwörter für Aggregation (Zählung, Statistik)
    agg_words = ["anzahl", "wie viele", "meisten", "höchsten", "durchschnitt", "summe", "top", "gesamt"]

    # Schlüsselwörter für Listen (Aufzählungen, Inhalte)
    list_words = ["welche", "liste", "zeige", "nenne", "gib", "tracklist", "songs", "titel", "alben", "kunden"]

    if any(w in ql for w in agg_words):
        return "aggregate"
    if any(w in ql for w in list_words):
        return "list"

    # Fallback: lieber Liste als zählen
    return "list"

# -----------------------------------
# Prompt-Vorlagen
# -----------------------------------
SYSTEM_AGG = """
You are a SQL expert for the Chinook schema (PostgreSQL). Use these aliases:
Artist AS ar(ArtistId, Name), Album AS al(AlbumId, Title, ArtistId),
Track AS tr(TrackId, Name, AlbumId, GenreId), Genre AS ge(GenreId, Name).

Task: Return ONE plain SQL SELECT (no Markdown, no explanations).
Goal: Aggregation (who/what has MOST/COUNT).
Rules:
- Select a descriptive text column FIRST (e.g., ar.Name AS name)
- Select a numeric aggregation SECOND (e.g., COUNT(*) AS anzahl)
- ORDER BY anzahl DESC
- LIMIT {top_k}
- Use only columns that exist in the provided schema info.
- Prefer joins like: ar JOIN al ON ar.ArtistId = al.ArtistId
- Never reference t1.name etc. if that alias/column doesn't exist.
"""

SYSTEM_LIST = """
You are a SQL expert for the Chinook schema (PostgreSQL). Use these aliases:
Artist AS ar(ArtistId, Name), Album AS al(AlbumId, Title, ArtistId),
Track AS tr(TrackId, Name, AlbumId, GenreId), Genre AS ge(GenreId, Name).

Task: Return ONE plain SQL SELECT (no Markdown, no explanations).
Goal: Listing (return rows, no COUNT unless asked).
Rules:
- Choose meaningful columns (e.g., al.Title AS album, ar.Name AS artist)
- Use joins like: ar JOIN al ON ar.ArtistId = al.ArtistId
- Use WHERE filters based on the question.
- ORDER BY a relevant column (e.g., album)
- LIMIT {top_k} unless the user explicitly asks for all.
- Never reference t1.name etc. if that alias/column doesn't exist.
"""

prompt_agg = ChatPromptTemplate([
    ("system", SYSTEM_AGG),
    ("user",   "Question: {input}\nDialect: {dialect}\nTables:\n{table_info}")
])

prompt_list = ChatPromptTemplate([
    ("system", SYSTEM_LIST),
    ("user",   "Question: {input}\nDialect: {dialect}\nTables:\n{table_info}")
])

# -----------------------------------
# Hilfsfunktionen
# -----------------------------------
def extract_sql(text: str) -> str:
    m = re.search(r"```sql(.*?)```", text, flags=re.I|re.S)
    if m:
        sql = m.group(1).strip()
    else:
        m = re.search(r"(SELECT[\s\S]+)$", text, flags=re.I)
        sql = m.group(1).strip() if m else text.strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql

def generate_query(question: str, top_k: int = 10):
    intent = detect_intent(question)
    if intent == "list":
        system_prefix = (
            "TASK: You must return a LISTING query — do NOT use COUNT, SUM, AVG, GROUP BY, or aggregation.\n"
            "Focus on selecting text columns like track.name, album.title, artist.name, etc.\n"
        )
        template = prompt_list
    else:
        system_prefix = (
            "TASK: You must return an AGGREGATION query using COUNT, SUM or similar, "
            "grouped and ordered appropriately.\n"
        )
        template = prompt_agg

    prompt = template.invoke({
        "dialect": db.dialect,
        "top_k": top_k,
        "table_info": db.get_table_info(),
        "input": system_prefix + question,   # Intent wird in den Text injiziert
    })
    response = model.invoke(prompt)
    return extract_sql(response.content)

def execute_query(sql: str, question: str = ""):
    """
    Führt die generierte SQL-Abfrage robust aus:
    - normalisiert Spaltennamen (ArtistId → artist_id)
    - macht Textvergleiche tolerant (= → ILIKE '%...%')
    - versucht bei Fehlern eine automatische Korrektur über das Modell
    """
    tool = QuerySQLDatabaseTool(db=db)

    # -----------------------------------
    # 1️⃣ Spaltennamen-Fix: CamelCase → snake_case
    # -----------------------------------
    def fix_id_names(text):
        patterns = {
            "artistid": "artist_id",
            "albumid": "album_id",
            "customerid": "customer_id",
            "employeeid": "employee_id",
            "trackid": "track_id",
            "genreid": "genre_id",
            "invoiceid": "invoice_id",
            "invoicelineid": "invoice_line_id",
            "playlistid": "playlist_id",
        }
        for wrong, right in patterns.items():
            text = re.sub(rf"\b{wrong}\b", right, text, flags=re.I)
        return text

    # -----------------------------------
    # 2️⃣ Vergleichsoperator-Fix: "=" → ILIKE '%...%'
    # -----------------------------------
    def soften_equals(text):
        """
        Sucht in WHERE-Bedingungen nach Gleichheitsvergleichen mit Text
        und wandelt sie in eine case-insensitive Teilstring-Suche um.
        """
        def repl(m):
            left, val = m.group(1), m.group(2)
            if not val.startswith("'") or "%" in val:
                return m.group(0)  # bereits ein Pattern oder kein String
            return f"{left} ILIKE '%' || {val.strip()} || '%'"
        return re.sub(r"(\b[a-z_\.]+\b)\s*=\s*('.*?')", repl, text, flags=re.I)

    # -----------------------------------
    # 3️⃣ SQL vorbereiten
    # -----------------------------------
    sql_fixed = fix_id_names(sql)
    sql_fixed = soften_equals(sql_fixed)
    sql_fixed = re.sub(r'\bAS\b', 'as', sql_fixed, flags=re.I)

    # -----------------------------------
    # 4️⃣ Erste Ausführung
    # -----------------------------------
    try:
        return tool.invoke(sql_fixed)

    # -----------------------------------
    # 5️⃣ Fehler → Retry mit LLM-Korrektur
    # -----------------------------------
    except Exception as e1:
        st.warning("🔁 Fehler erkannt – versuche automatische SQL-Korrektur …")

        fix_msg = f"""The following SQL caused an error in PostgreSQL:

SQL:
{sql_fixed}

ERROR:
{str(e1)}

Please return ONLY a corrected SQL query valid for the Chinook (PostgreSQL) schema.
Use snake_case for all column and alias names (e.g. artist_id, album_id).
Avoid quoting or COUNT(*) unless needed."""
        resp = model.invoke(fix_msg)
        fixed_sql = extract_sql(resp.content)
        fixed_sql = fix_id_names(fixed_sql)
        fixed_sql = soften_equals(fixed_sql)

        try:
            return tool.invoke(fixed_sql)
        except Exception as e2:
            st.error(f"Fehler bei der SQL-Ausführung: {e2}")
            st.code(fixed_sql, language="sql")
            return []

def render_answer(result):
    """Formuliert eine verständliche Antwort, egal ob 1 oder 2 Spalten."""
    if not result:
        return "Keine Ergebnisse gefunden."
    if isinstance(result, str):
        try:
            result = ast.literal_eval(result)
        except Exception:
            return f"Rohdaten: {result}"
    if not isinstance(result, (list, tuple)):
        result = [result]
    if result and not isinstance(result[0], (list, tuple)):
        result = [(r,) for r in result]

    df = pd.DataFrame(result)
    if df.shape[1] == 1:
        vals = df.iloc[:, 0].astype(str).tolist()
        return f"Gefundene Werte: {', '.join(vals)}"
    else:
        a, b = df.iloc[0, 0], df.iloc[0, 1]
        is_num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        if is_num(a) and not is_num(b):
            name, count = str(df.iloc[0, 1]), df.iloc[0, 0]
        elif is_num(b) and not is_num(a):
            name, count = str(df.iloc[0, 0]), df.iloc[0, 1]
        else:
            name, count = str(df.iloc[0, 0]), str(df.iloc[0, 1])
        return f"Am häufigsten: **{name}** mit **{count} Einträgen.**"

# -----------------------------------
# Streamlit Frontend
# -----------------------------------
st.set_page_config(page_title="LangChain SQL-Q&A", page_icon="🧠", layout="wide")
st.title("🧠 SQL-Frage-Antwort-System (LangChain + Chinook)")

user_input = st.text_input("Frage an die Datenbank:", "Welche Künstler haben die meisten Alben veröffentlicht?")
if st.button("Frage ausführen"):
    if not user_input.strip():
        st.warning("Bitte zuerst eine Frage eingeben.")
    else:
        with st.spinner("Verarbeite Frage..."):
            sql_query = generate_query(user_input)
            result = execute_query(sql_query, user_input)
            answer = render_answer(result)

        st.subheader("🧾 Generierte SQL-Abfrage")
        st.code(sql_query, language="sql")

        st.subheader("💬 Antwort")
        st.success(answer)
        

                # -----------------------------------
        # Ergebnis-Darstellung (intelligent)
        # -----------------------------------
        if result:
            if isinstance(result, str):
                try:
                    result = ast.literal_eval(result)
                except Exception:
                    st.error("Das Abfrageergebnis konnte nicht geparst werden.")
                    result = []

            # Immer Listen/Tupel-Struktur erzwingen
            if not isinstance(result, (list, tuple)):
                result = [result]
            if result and not isinstance(result[0], (list, tuple)):
                result = [(result[0],)]

            df = pd.DataFrame(result)

            # Dynamische Darstellung:
            # 1️⃣ reine Listen (Textausgabe)
            if df.shape[1] == 1 or (
                df.shape[1] > 1
                and not any(isinstance(x, (int, float)) for x in df.iloc[0])
            ):
                st.subheader("📜 Ergebnisliste")
                for _, row in df.iterrows():
                    st.markdown("- " + ", ".join(map(str, row.values)))

            # 2️⃣ Aggregation (z. B. Name + Anzahl)
            elif df.shape[1] == 2:
                a, b = df.iloc[0, 0], df.iloc[0, 1]
                is_num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
                if is_num(a) and not is_num(b):
                    df = df[[1, 0]]
                df.columns = ["Name", "Anzahl"]
                st.subheader("📊 Abfrageergebnis")
                st.dataframe(df, use_container_width=True)

            # 3️⃣ Mehrspaltige Ergebnisse (z. B. Track, Album, Artist)
            else:
                df.columns = [f"Spalte_{i+1}" for i in range(df.shape[1])]
                st.subheader("📊 Mehrspaltiges Ergebnis")
                st.dataframe(df, use_container_width=True)