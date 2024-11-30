import numpy as np
from abc import abstractmethod
import random
from policy import Policy

def Policy2210xxx(Policy):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)


class Policy2210xxx(Policy):
    def __init__(self):
        self.parts = None  # Will be initialized dynamically
        self.population = None  # Will be initialized dynamically
        self.trace = {"mintrace": [], "bestpop": []}

    def _initialize_parts_(self, products):
        # Convert environment products into parts
        return [
            {"height": product["size"][1], "width": product["size"][0], "count": product["quantity"]}
            for product in products
        ]
        
    def setup(self, products, max_height):
        # Initialize parts based on products and environment constraints
        self.parts = [{"height": p["size"][1], "width": p["size"][0], "count": p["quantity"]} for p in products]
        self.max_height = max_height
        
        # Check feasibility before proceeding
        min_required_height = sum(p["height"] for p in self.parts)
        if min_required_height > max_height:
            raise ValueError("Infeasible problem: Total required height exceeds max_height.")
        self.population = self._initialize_population_()
        
    def _initialize_population_(self):
        NIND = 100  # Number of individuals
        population = np.zeros((NIND, len(self.parts)), dtype=int)

        for i in range(NIND):
            attempts = 0
            while attempts < 100:  # Limit retries to prevent infinite loop
                attempts += 1
                part_counts = np.random.randint(0, [part["count"] + 1 for part in self.parts])
                total_height = sum(part_counts[j] * self.parts[j]["height"] for j in range(len(self.parts)))

                # If total height satisfies constraints, add to population
                if total_height <= self.max_height:
                    population[i, :] = part_counts
                    break
        
        # If all attempts fail, fallback to a minimal valid chromosome
        if attempts == 100:
            print(f"Warning: Individual {i} could not satisfy max_height. Using fallback.")
            for j in range(len(self.parts)):
                max_count = self.max_height // self.parts[j]["height"]
                population[i, j] = min(max_count, self.parts[j]["count"])

        return population

    def get_action(self, observation, info):
        cost, fitness = self._calculate_fitness_(self.population)

        # Select the best chromosome
        best_idx = np.argmin(cost)
        best_chromosome = self.population[best_idx, :]

        for j, count in enumerate(best_chromosome):
            if count > 0:
            	prod_size = [self.parts[j]["width"], self.parts[j]["height"]]
            
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                if stock_w < prod_w or stock_h < prod_h:
                    continue

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        return {"stock_idx": i, "size": prod_size, "position": (x, y)}
            return {"stock_idx": 0, "size": [1, 1], "position": [0, 0]}
      
