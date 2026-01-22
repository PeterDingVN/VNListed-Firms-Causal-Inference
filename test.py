import re
import pandas as pd

pattern = r'Q[1-4]\.\d{4}$'

# Test examples
test_cases = [
    'Quy hang namw: Q1.2015',  # Match
    'Q2.2016',  # Match
    'Q3.2024',  # Match
    'Q4.2025',  # Match
    'Q5.2015',  # No match (5 is not valid)
    'Q0.2015',  # No match (0 is not valid)
    'Q1.202',   # No match (only 3 digits for year)
    'Q1.20155', # No match (5 digits for year)
    'q1.2015',  # No match (lowercase)
]

for test in test_cases:
    if not re.search(pattern, test):

    else:
        print(f"✗ {test}")