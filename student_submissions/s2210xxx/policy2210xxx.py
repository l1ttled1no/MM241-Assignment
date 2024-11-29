from policy import Policy
import random
from abc import abstractmethod
import numpy as np

from policy import Policy
import random
from abc import abstractmethod
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self):
        super().__init__()
        # Student code here
        pass

    # Student code here
    # You can add more functions if needed

    def get_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]
        prod_height = []
        prod_width = []
        prod_quantity = []

        # Initialize arrays to store stock sizes
        stock_sizes = []

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Store the sizes of products in array
                prod_height.append(int(prod_h))
                prod_width.append(int(prod_w))
                prod_quantity.append(int(prod["quantity"]))

        # Loop through all stocks to get their sizes
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_sizes.append((int(stock_w), int(stock_h)))

        # Print the arrays
        print("Product Heights:", prod_height)
        print("Product Widths:", prod_width)
        print("Product Quantities:", prod_quantity)

        # Print the number of stocks and their sizes
        print("Number of Stocks:", len(observation["stocks"]))
        print("Stock Sizes:", stock_sizes)

        actions = []


        for prod_idx in range(len(prod_height)):
            # H = stock_sizes[prod_idx][0]
            # W = stock_sizes[prod_idx][1]
            # print("H = ",H,"W = ",W)
            stock_actions = self.solve_cutting_stock(prod_height,prod_width,prod_idx,observation)
            actions.extend(stock_actions)
        # Return the first action from the generated actions
        
        return actions
    
    def solve_cutting_stock(self, h, w, stock_index,observation):
        # Initialize the array to store the number of item type
        result = [0] * len(h)
        # Sort items through value density (value / area)
        v = [0] * len(h)
        v[stock_index] = 1
        #print(v)
        items = sorted(
            zip(w,h, v, range(len(h))), key=lambda x: x[2] / (x[0] * x[1]), reverse=True
        )
        #print(items)
        actions = []
        #stock = (H, W)
        stock = observation["stocks"][stock_index]
        stock_w, stock_h = self._get_stock_size_(stock)
        W, H = self._get_stock_size_(stock)
        #print(H," ",W)
        current_y = 0  # Keep track of the current height position globally

        for width, height, value, index in items:
            x, y = 0, current_y  # Start from the current height position

            # Calculate the maximum number of items that can fit in the remaining stock
            remaining_H = H - current_y  # Calculate remaining height
            max_item_h = remaining_H // height
            max_item_w = W // width
            max_count = max_item_h * max_item_w

            if max_count > 0:
                # Update array
                result[index] += int(max_count)
                
                # Place items row by row
                for row in range(max_item_h):
                    for col in range(max_item_w):
                        prod_size = [width, height]
                        if self._can_place_(stock, (x, y), prod_size):
                            actions.append({
                                "stock_idx": stock_index,
                                "size": prod_size,
                                "position": (x, y)
                            })
                        x += width  # Move right for next item in row
                    y += height    # Move down for next row
                    x = 0         # Reset x position for new row

                # Update the current_y position for the next item type
                current_y = y

                # Update remaining stock dimensions
                H -= height * max_item_h
                W -= width * max_item_w

            # If the stock is full, stop placing items
            if H <= 0 or W <= 0:
                break
                
                #print(actions)
        print(result)   
        return actions



