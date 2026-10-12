from src.evaluation.evaluator import evaluate_answer


question = """
What are the eligibility criteria under PMAY-U 2.0?
"""


answer = """
The eligibility criteria for PMAY-U 2.0 include:

1. Construction of a house within the stipulated time period
as mentioned in the Agreement under the PMAY-U 2.0 Scheme.

2. Not selling or transferring the house for which subsidy
has been received under the Interest Subsidy Scheme for
five years from the date of release of the first instalment.

3. Using the house only for residential purposes.

4. Adhering to all terms and conditions of the PMAY-U 2.0 Scheme.

5. Not applying or availing benefit under another vertical
of PMAY-U 2.0.
"""


context = """
The beneficiary shall construct the house within the
stipulated time period as mentioned in the Agreement
under the PMAY-U 2.0 Scheme.

The beneficiary shall not sell or transfer the house
for which subsidy has been received under ISS vertical
of PMAY-U 2.0 Scheme up to a period of Five (5) years
from the date of release of the first instalment.

The beneficiary shall use the house constructed under
PMAY-U 2.0 Scheme only for residential purposes.

The beneficiary shall adhere to all terms and conditions
of the PMAY-U 2.0 Scheme.

The beneficiary shall not apply or avail benefit under
any other vertical of PMAY-U 2.0.
"""


result = evaluate_answer(
    question,
    answer,
    context
)


print("\n")
print("=" * 60)
print("EVALUATOR TEST")
print("=" * 60)

print(result)