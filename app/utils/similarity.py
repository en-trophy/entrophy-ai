def compare_feature(user: dict, answer: dict) -> float:
    total = 0
    matched = 0

    def compare_dict(u, a):
        nonlocal total, matched
        for k in a:
            if isinstance(a[k], dict):
                compare_dict(u.get(k, {}), a[k])
            else:
                total += 1
                if u.get(k) == a[k]:
                    matched += 1

    compare_dict(user, answer)
    return matched / total if total else 0.0
