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

import itertools as it
import logpy.core as lc
from sympy.ntheory.generate import prime, isprime
import logpy

# Check if the elements of x are prime 
def check_prime(x):
    if lc.isvar(x):
        return lc.condeseq([(lc.eq, x, p)] for p in map(prime, it.count(1)))
    else:
        return lc.success if isprime(x) else lc.fail

# Declate the variable
x = lc.var()

# Check if an element in the list is a prime number
list_nums = (23, 4, 27, 17, 13, 10, 21, 29, 3, 32, 11, 19)
print('\nList of primes in the list:')
print(set(lc.run(0, x, (logpy.membero, x, list_nums), (check_prime, x))))

# Print first 7 prime numbers
print('\nList of first 7 prime numbers:')
print(lc.run(7, x, check_prime(x)))
