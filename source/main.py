from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Cargamos la bdd
sqlite_uri = "sqlite:///./databases/Chinook_sample.db"
db = SQLDatabase.from_uri(sqlite_uri)

# Plantilla del prompt
template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)


def get_schema(_):
    schema = db.get_table_info()
    return schema


llm = ChatOllama(model="gemma:2b", temperature=0)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


user_question = "how many albums are there in the database?"
sql_chain.invoke({"question": user_question})

# 'SELECT COUNT(*) AS TotalAlbums\nFROM Album;'
