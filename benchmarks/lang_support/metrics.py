"""Language-independent ASR scoring."""


def levenshtein_ops(reference, hypothesis):
    """Return substitutions, deletions, insertions, and reference length."""
    n, m = len(reference), len(hypothesis)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    operation = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        distance[i][0], operation[i][0] = i, "D"
    for j in range(1, m + 1):
        distance[0][j], operation[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j], operation[i][j] = distance[i - 1][j - 1], "E"
            else:
                # Rank keys break distance ties toward substitution, then
                # insertion, so the S/D/I split stays stable across runs
                _, _, operation[i][j] = best = min(
                    (distance[i - 1][j - 1] + 1, 0, "S"),
                    (distance[i][j - 1] + 1, 1, "I"),
                    (distance[i - 1][j] + 1, 2, "D"),
                )
                distance[i][j] = best[0]
    i, j = n, m
    substitutions = deletions = insertions = 0
    while i or j:
        op = operation[i][j]
        if op == "E":
            i, j = i - 1, j - 1
        elif op == "S":
            substitutions, i, j = substitutions + 1, i - 1, j - 1
        elif op == "I":
            insertions, j = insertions + 1, j - 1
        elif op == "D":
            deletions, i = deletions + 1, i - 1
        else:
            raise AssertionError(f"No edit operation at {(i, j)}")
    return substitutions, deletions, insertions, n


def wer_details(reference, hypothesis, normalize):
    """Return WER, S, D, I, N, and accuracy after normalization."""
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    s, d, i, n = levenshtein_ops(r, h)
    errors = s + d + i
    return (errors / n if n else float("nan"), s, d, i, n,
            (n - errors) / n if n else float("nan"))


def cer(reference, hypothesis, normalize):
    """Return character error rate after normalization."""
    r = list(normalize(reference).replace(" ", ""))
    h = list(normalize(hypothesis).replace(" ", ""))
    s, d, i, n = levenshtein_ops(r, h)
    return (s + d + i) / n if n else float("nan")
