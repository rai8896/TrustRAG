from src.retrieval.query_expansion import expand_query


question = input(
    "Enter question: "
)

queries = expand_query(
    question
)

print(
    "\n"
    + "=" * 70
)

print(
    "EXPANDED QUERIES"
)

print(
    "=" * 70
)

for i, query in enumerate(
    queries,
    start=1
):

    print(
        f"{i}. {query}"
    )