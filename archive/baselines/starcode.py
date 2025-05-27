from typing import List
from collections import deque

def levenshtein(s1: str, s2: str) -> int:

    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = previous_row[j + 1] + 1
            delete = current_row[j] + 1
            substitute = previous_row[j] + (c1 != c2)
            current_row.append(min(insert, delete, substitute))
        previous_row = current_row

    return previous_row[-1]


def starcode(strings: List[str], max_distance: int = 1) -> List[List[str]]:

    unvisited = set(strings)
    clusters = []

    while unvisited:
        # Pick a new center
        center = unvisited.pop()
        cluster = [center]
        queue = deque([center])

        while queue:
            current = queue.popleft()
            to_check = list(unvisited)
            for s in to_check:
                if levenshtein(current, s) <= max_distance:
                    unvisited.remove(s)
                    cluster.append(s)
                    queue.append(s)

        clusters.append(cluster)

    return clusters
