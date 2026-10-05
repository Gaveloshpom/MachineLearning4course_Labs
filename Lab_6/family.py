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

import json
from logpy import Relation, facts, run, conde, var, eq

# Check if 'x' is the parent of 'y'
def parent(x, y):
    return conde([father(x, y)], [mother(x, y)])

# Check if 'x' is the grandparent of 'y'
def grandparent(x, y):
    temp = var()
    return conde((parent(x, temp), parent(temp, y)))

# Check for sibling relationship between 'a' and 'b'  
def sibling(x, y):
    temp = var()
    return conde((parent(temp, x), parent(temp, y)))

# Check if x is y's uncle
def uncle(x, y):
    temp = var()
    return conde((father(temp, x), grandparent(temp, y)))

if __name__=='__main__':
    father = Relation()
    mother = Relation()
    
    with open('relationships.json') as f:
        d = json.loads(f.read())

    for item in d['father']:
        facts(father, (list(item.keys())[0], list(item.values())[0]))

    for item in d['mother']:
        facts(mother, (list(item.keys())[0], list(item.values())[0]))

    x = var()

    # John's children
    name = 'John'
    output = run(0, x, father(name, x))
    print("\n---List of " + name + "'s children:")
    for item in output:
        print(item, end=' ')

    # William's mother
    name = 'William'
    output = run(0, x, mother(x, name))[0]
    print("\n\n" + name + "'s mother:\n" + output)

    # Adam's parents 
    name = 'Adam'
    output = run(0, x, parent(x, name))
    print("\n\n---List of " + name + "'s parents:")
    for item in output:
        print(item, end=' ')

    # Wayne's grandparents 
    name = 'Wayne'
    output = run(0, x, grandparent(x, name))
    print("\n\n---List of " + name + "'s grandparents:")
    for item in output:
        print(item, end=' ')

    # Megan's grandchildren 
    name = 'Megan'
    output = run(0, x, grandparent(name, x))
    print("\n\n---List of " + name + "'s grandchildren:")
    for item in output:
        print(item, end=' ')

    # David's siblings 
    name = 'David'
    output = run(0, x, sibling(x, name))
    siblings = [x for x in output if x != name]
    print("\n\n---List of " + name + "'s siblings:")
    for item in siblings:
        print(item, end=' ')

    # Tiffany's uncles
    name = 'Tiffany'
    name_father = run(0, x, father(x, name))[0]
    output = run(0, x, uncle(x, name))
    output = [x for x in output if x != name_father]
    print("\n\nList of " + name + "'s uncles:")
    for item in output:
        print(item, end=' ')

    # All spouses
    a, b, c = var(), var(), var()
    output = run(0, (a, b), (father, a, c), (mother, b, c))
    print("\n\nList of all spouses:")
    for item in output:
        print('Husband:', item[0], '<==> Wife:', item[1])
     
