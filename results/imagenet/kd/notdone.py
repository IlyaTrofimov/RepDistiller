import os
import sys

done = set(range(0, 100))

for elem in os.listdir('.'):
    if elem.endswith('.json'):
        elem = elem.replace('part_27_', '').replace('_history.json', '')

        done.remove(int(elem))

print(done)
