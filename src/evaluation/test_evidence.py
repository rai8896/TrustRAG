from evidence_checker import check_evidence


answer = """
Families belonging to EWS, LIG and MIG categories
living in urban areas without a pucca house are eligible.
EWS households have annual income up to ₹3 lakh.
Applicants must be Indian citizens for at least 10 years.
"""


context = """
3.1 Families belonging to EWS/LIG/MIG category,
living in urban areas, having no pucca house
anywhere in the country, are eligible to purchase
or construct a house under PMAY-U 2.0.

3.3 EWS households are defined as households
with an annual income of up to ₹3 lakh.
"""


result = check_evidence(
    answer,
    context
)


print("\n" + "=" * 60)
print("EVIDENCE CHECK")
print("=" * 60)

print(result)