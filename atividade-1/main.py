from math import sqrt


ratings = {
    "Alice": {"item1": 5, "item2": 3, "item3": 4, "item4": 4, "item5": None},
    "User1": {"item1": 3, "item2": 1, "item3": 2, "item4": 3, "item5": 3},
    "User2": {"item1": 4, "item2": 3, "item3": 4, "item4": 3, "item5": 5},
    "User3": {"item1": 3, "item2": 3, "item3": 1, "item4": 5, "item5": 4},
    "User4": {"item1": 1, "item2": 5, "item3": 5, "item4": 2, "item5": 1},
}


def average(user_ratings):
    rated = [rating for rating in user_ratings.values() if rating is not None]
    return sum(rated) / len(rated)


def pearson(user_a, user_b):
    """Pearson correlation over items rated by both users."""
    common_items = [
        item
        for item in user_a
        if user_a[item] is not None and user_b[item] is not None
    ]
    if len(common_items) < 2:
        return 0.0

    mean_a = sum(user_a[item] for item in common_items) / len(common_items)
    mean_b = sum(user_b[item] for item in common_items) / len(common_items)
    numerator = sum(
        (user_a[item] - mean_a) * (user_b[item] - mean_b)
        for item in common_items
    )
    denominator_a = sqrt(sum((user_a[item] - mean_a) ** 2 for item in common_items))
    denominator_b = sqrt(sum((user_b[item] - mean_b) ** 2 for item in common_items))
    denominator = denominator_a * denominator_b
    return numerator / denominator if denominator else 0.0


def predict(user_name, item, ratings, similarities, neighbors):
    """Mean-centered weighted prediction using the selected neighbors."""
    user_mean = average(ratings[user_name])
    selected = [
        (other, similarities[other])
        for other in neighbors
        if ratings[other][item] is not None and similarities[other] > 0
    ]
    similarity_sum = sum(similarity for _, similarity in selected)
    if not selected or similarity_sum == 0:
        raise ValueError(f"No positive-similarity neighbors rated {item}.")

    weighted_deviations = sum(
        similarity * (ratings[other][item] - average(ratings[other]))
        for other, similarity in selected
    )
    return user_mean + weighted_deviations / similarity_sum


if __name__ == "__main__":
    target = "Alice"
    similarities = {
        user: pearson(ratings[target], user_ratings)
        for user, user_ratings in ratings.items()
        if user != target
    }

    print("Similaridade de Pearson com Alice:")
    for user, similarity in similarities.items():
        print(f"  {user}: {similarity:.2f}")

    # O exemplo do livro escolhe User1 e User2 como vizinhos de Alice.
    neighbors = ["User1", "User2"]
    prediction = predict(target, "item5", ratings, similarities, neighbors)
    print(f"\nPrevisão de Alice para item5: {prediction:.2f} / 5")
