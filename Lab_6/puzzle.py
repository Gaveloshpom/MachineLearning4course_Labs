# --- compatibility shim (place at top of your script, BEFORE importing logpy) ---
import collections
try:
    # Python 3.3+ correct place
    from collections.abc import Iterator
except ImportError:
    # fallback for very old Pythons (rare)
    from collections import Iterator

# if the package does "from collections import Iterator" we provide attribute
if not hasattr(collections, 'Iterator'):
    collections.Iterator = Iterator
# ------------------------------------------------------------------------------

from logpy import *
from logpy.core import lall

# Declare the variable
people = var()

# Define the rules
rules = lall(
    # There are 4 people
    (eq, (var(), var(), var(), var()), people),

    # Steve's car is blue
    (membero, ('Steve', var(), 'blue', var()), people),

    # Person who owns the cat lives in Canada 
    (membero, (var(), 'cat', var(), 'Canada'), people),

    # Matthew lives in USA
    (membero, ('Matthew', var(), var(), 'USA'), people),

    # The person who has a black car lives in Australia
    (membero, (var(), var(), 'black', 'Australia'), people),

    # Jack has a cat
    (membero, ('Jack', 'cat', var(), var()), people),

    # Alfred lives in Australia
    (membero, ('Alfred', var(), var(), 'Australia'), people),

    # Person who owns the dog lives in France 
    (membero, (var(), 'dog', var(), 'France'), people),

    # Who is the owner of the rabbit? 
    (membero, (var(), 'rabbit', var(), var()), people)
)

# Run the solver
solutions = run(0, people, rules)

# Extract the output
output = [house for house in solutions[0] if 'rabbit' in house][0][0]

# Print the output
print('\n' + output + ' is the owner of the rabbit')
print('\nHere are all the details:')
attribs = ['Name', 'Pet', 'Color', 'Country']
print('\n' + '\t\t'.join(attribs))
print('=' * 57)
for item in solutions[0]:
    print('')
    print('\t\t'.join([str(x) for x in item]))

