import collections
from typing import List

from sortedcontainers import SortedSet


# 2353. Design a Food Rating System
class FoodRatings:
    def __init__(self, foods: List[str], cuisines: List[str],
                 ratings: List[int]):
        self.dict_cuisine_rating_and_food = collections.defaultdict(
            lambda: SortedSet(
                key=lambda rating_food: (-rating_food[0], rating_food[1])
            )
        )
        self.dict_food_cuisine = {}
        self.dict_food_rating = {}

        for food, cuisine, rating in zip(foods, cuisines, ratings):
            self.dict_cuisine_rating_and_food[cuisine].add((rating, food))
            self.dict_food_cuisine[food] = cuisine
            self.dict_food_rating[food] = rating

    def changeRating(self, food: str, rating: int) -> None:
        cuisine = self.dict_food_cuisine[food]
        rating_food_in_cuisine = self.dict_cuisine_rating_and_food[cuisine]
        old_rating = self.dict_food_rating[food]

        rating_food_in_cuisine.remove((old_rating, food))
        rating_food_in_cuisine.add((rating, food))
        self.dict_food_rating[food] = rating

    def highestRated(self, cuisine: str) -> str:
        return self.dict_cuisine_rating_and_food[cuisine][0][1]
