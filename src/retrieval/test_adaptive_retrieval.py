from src.retrieval.adaptive_retrieval import (
    get_initial_top_k,
    increase_top_k
)


print("=" * 60)
print("ADAPTIVE RETRIEVAL TEST")
print("=" * 60)


# LOW
result = get_initial_top_k("LOW")

print("\nLOW RISK")
print(result)


# MEDIUM
result = get_initial_top_k("MEDIUM")

print("\nMEDIUM RISK")
print(result)


# HIGH
result = get_initial_top_k("HIGH")

print("\nHIGH RISK")
print(result)


# Repair
print("\nREPAIR TEST")

top_k = 3

for i in range(4):

    print(
        f"TOP-K: {top_k}"
    )

    top_k = increase_top_k(
        top_k
    )