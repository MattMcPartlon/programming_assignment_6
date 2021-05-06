import numpy as np
from SkipList import SkipList
height, p = 10, 0.5
n_items = 200

#make a skip list with specified height and
#bubble-up probability
skip_list = SkipList(height=height, p=p)
items_to_add = np.random.randint(0, 2**(height+1), n_items)
items_to_add = [x for x in set(items_to_add)]
#add all the items
for item in items_to_add:
    skip_list.add(item)
    assert item in skip_list

assert skip_list.num_vals == len(set(items_to_add))

print('skip list size :',skip_list.num_vals)
print('skip list num elts:',skip_list.num_elts)
ratio = skip_list.num_elts/skip_list.num_vals
print('ratio : ',ratio)
print('actual num vals : ',len(items_to_add))

num_added = skip_list.num_vals

for i, to_remove in enumerate(items_to_add):
    skip_list.remove(to_remove)
    print('removed item :',to_remove)
    assert to_remove not in skip_list
    print('skip list size :',skip_list.num_vals)
    print('skip list num elts:',skip_list.num_elts)
    assert skip_list.num_vals == num_added - i -1
    for still_in in items_to_add[i+1:]:
        assert still_in in skip_list

