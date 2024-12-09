from policy import Policy
import random
from abc import abstractmethod
import numpy as np
from scipy.optimize import linprog

class Policy2210xxx(Policy):
    # def __init__(self):
    #     # Student code here
    #     self.H = []
    #     self.W = []
    #     self.h = []
    #     self.w = []
    #     self.demands = []
    #     self.pattern = []
    #     self.waste = []
    #     pass

    # # Student code here
    # # You can add more functions if needed

    # def find_dual_prices(self):
    #     """
    #     Solve the primal problem and extract dual prices.

    #     Minimize: x1 + x2 + x3
    #     Subject to: Ax >= demands

    #     Returns:
    #     array: The dual prices for each constraint
    #     """
    #     A = np.array(self.pattern)
    #     demands = np.array(self.demands)

    #     # print("\nInitial Pattern Matrix (A):")
    #     # print(A)
    #     # print("Demands:", demands)

    #     A_transposed = A.T
    #     # print("\nTransposed Pattern Matrix (A.T):")
    #     # print(A_transposed)

    #     c = np.ones(A.shape[0])  # Number of rows in A (number of patterns)

    #     res = linprog(c, A_ub=-A_transposed, b_ub=-demands, method='highs')

    #     if res.success:
    #         dual_prices = res.ineqlin.marginals
    #         negated_dual_prices = [-float(price) for price in dual_prices]
    #         return negated_dual_prices
    #     else:
    #         print("Optimization failed:", res.message)
    #         return None

    # def get_action(self, observation, info):
    #     # Student code here
    #     list_prods = observation["products"]
        
    #     # Clear previous values
    #     self.h = []
    #     self.w = []
    #     self.demands = []

    #     for prod in list_prods:
    #         if prod["quantity"] > 0:
    #             prod_size = prod["size"]
    #             prod_w, prod_h = prod_size

    #             # Store the sizes of products in class variables
    #             self.h.append(int(prod_h))
    #             self.w.append(int(prod_w))
    #             self.demands.append(int(prod["quantity"]))

    #     # Store stock sizes
    #     stock_sizes = []
    #     for stock in observation["stocks"]:
    #         stock_w, stock_h = self._get_stock_size_(stock)
    #         stock_sizes.append((int(stock_w), int(stock_h)))

    #     # Print for verification
    #     print("Product Heights (h):", self.h)
    #     print("Product Widths (w):", self.w)
    #     print("Product Quantities (demands):", self.demands)
    #     print("Number of Stocks:", len(observation["stocks"]))
    #     print("Stock Sizes:", stock_sizes)
        
    #     actions = []  # This will be our 2D array
    #     # Generate initial patterns
    #     for stock_idx in range(len(self.h)):
    #         # Initialize with simple patterns
    #         stock_actions, result = self.solve_cutting_stock(self.h, self.w, stock_idx, observation)
    #         actions.append(stock_actions)
    #         self.pattern.append(result)
        
    #     # Column generation process
    #     iteration = 0
    #     max_iterations = 100  # Prevent infinite loops
        
    #     while iteration < max_iterations:
    #         # Get dual prices
    #         dual_prices = self.find_dual_prices()
    #         if dual_prices is None:
    #             break
                
    #         print(f"\nIteration {iteration} - Dual Prices:", dual_prices)
            
    #         # Try to generate new pattern for each stock size
    #         best_reduced_cost = 0
    #         best_pattern = None
    #         best_actions = None
    #         best_stock_idx = None
            
    #         for stock_idx in range(len(observation["stocks"])):
    #             new_actions, new_pattern = self.solve_cutting_stock_skyline(
    #                 self.h, self.w, dual_prices, stock_idx, observation
    #             )
                
    #             # Calculate reduced cost
    #             reduced_cost = 1 - sum(new_pattern[i] * dual_prices[i] for i in range(len(self.h)))
                
    #             if reduced_cost < best_reduced_cost:
    #                 best_reduced_cost = reduced_cost
    #                 best_pattern = new_pattern
    #                 best_actions = new_actions
    #                 best_stock_idx = stock_idx
            
    #         # If no improving pattern found, stop
    #         if best_reduced_cost >= -1e-10:
    #             print("No improving pattern found. Stopping.")
    #             break
                
    #         # Add best pattern to patterns
    #         print(f"Adding new pattern with reduced cost: {best_reduced_cost}")
    #         self.pattern.append(best_pattern)
    #         if best_stock_idx < len(actions):
    #             actions[best_stock_idx].extend(best_actions)
    #         else:
    #             actions.append(best_actions)
                
    #         iteration += 1

    #     return actions
    
    # def solve_cutting_stock(self, h, w, stock_index,observation):
    #     # Initialize the array to store the number of item type
    #     result = [0] * len(h)
    #     # Sort items through value density (value / area)
    #     v = [0] * len(h)
    #     v[stock_index] = 1
    #     #print(v)
    #     items = sorted(
    #         zip(w,h, v, range(len(h))), key=lambda x: x[2] / (x[0] * x[1]), reverse=True
    #     )
    #     #print(items)
    #     actions = []
    #     #stock = (H, W)
    #     stock = observation["stocks"][stock_index]
    #     stock_w, stock_h = self._get_stock_size_(stock)
    #     W, H = self._get_stock_size_(stock)
    #     #print(H," ",W)
    #     current_y = 0  # Keep track of the current height position globally

    #     for width, height, value, index in items:
    #         x, y = 0, current_y  # Start from the current height position

    #         # Calculate the maximum number of items that can fit in the remaining stock
    #         remaining_H = H - current_y  # Calculate remaining height
    #         max_item_h = remaining_H // height
    #         max_item_w = W // width
    #         max_count = max_item_h * max_item_w

    #         if max_count > 0:
    #             # Update array
    #             result[index] += int(max_count)
                
    #             # Place items row by row
    #             for row in range(max_item_h):
    #                 for col in range(max_item_w):
    #                     prod_size = [width, height]
    #                     if self._can_place_(stock, (x, y), prod_size):
    #                         actions.append({
    #                             "stock_idx": stock_index,
    #                             "size": prod_size,
    #                             "position": (x, y)
    #                         })
    #                     x += width  # Move right for next item in row
    #                 y += height    # Move down for next row
    #                 x = 0         # Reset x position for new row

    #             # Update the current_y position for the next item type
    #             current_y = y

    #             # Update remaining stock dimensions
    #             H -= height * max_item_h
    #             W -= width * max_item_w

    #         # If the stock is full, stop placing items
    #         if H <= 0 or W <= 0:
    #             break
                
    #             #print(actions)
    #     #print(result)   
    #     return actions,result


    # def solve_cutting_stock_skyline(self, h, w, dual_prices, stock_index, observation):
    #     """
    #     Generate new cutting pattern using skyline packing and dual prices
    #     """
    #     class SkylineLevel:
    #         def __init__(self, x, y, width):
    #             self.x = x
    #             self.y = y
    #             self.width = width

    #     result = [0] * len(h)
    #     actions = []
    #     stock = observation["stocks"][stock_index]
    #     W, H = self._get_stock_size_(stock)
        
    #     # Initialize skyline with full width at y=0
    #     skyline = [SkylineLevel(0, 0, W)]
        
    #     # Sort items by (dual_price * width) / area for better packing
    #     items = [(i, w[i], h[i], dual_prices[i]) for i in range(len(h))]
    #     items.sort(key=lambda x: (x[3] * x[1]) / (x[1] * x[2]), reverse=True)

    #     placed = True
    #     while placed:
    #         placed = False
    #         for idx, width, height, dual_price in items:
    #             best_waste = float('inf')
    #             best_x = None
    #             best_y = None
    #             best_level_idx = None

    #             # Try to place item in each skyline level
    #             for i, level in enumerate(skyline):
    #                 if level.width >= width:
    #                     # Find the maximum height at this position
    #                     max_height = level.y
    #                     if max_height + height <= H:
    #                         waste = level.width - width
    #                         if waste < best_waste:
    #                             best_waste = waste
    #                             best_x = level.x
    #                             best_y = level.y
    #                             best_level_idx = i

    #             # If we found a valid position
    #             if best_x is not None:
    #                 # Place the item
    #                 if self._can_place_(stock, (best_x, best_y), [width, height]):
    #                     actions.append({
    #                         "stock_idx": stock_index,
    #                         "size": [int(width), int(height)],
    #                         "position": (int(best_x), int(best_y))
    #                     })
    #                     result[idx] += 1
    #                     placed = True

    #                     # Update skyline
    #                     level = skyline[best_level_idx]
    #                     if level.width == width:
    #                         level.y += height
    #                     else:
    #                         # Split the level
    #                         skyline[best_level_idx].width = level.width - width
    #                         skyline[best_level_idx].x = best_x + width
    #                         skyline.insert(best_level_idx, 
    #                                     SkylineLevel(best_x, best_y + height, width))
                            
    #                     # Merge adjacent levels of same height
    #                     i = 0
    #                     while i < len(skyline) - 1:
    #                         if skyline[i].y == skyline[i + 1].y:
    #                             skyline[i].width += skyline[i + 1].width
    #                             skyline.pop(i + 1)
    #                         else:
    #                             i += 1

    #     return actions, result

    # def __init__(self):
    #     self.patterns = {}
    #     self.current_pattern = None

    # def get_action(self, observation, info):
    #     products = observation["products"]
    #     stocks = observation["stocks"]

    #     # Find best cut considering demands and efficiency
    #     best_value = float('-inf')
    #     best_action = None

    #     # Track remaining space in current stock
    #     remaining_space = {}
    #     for i, stock in enumerate(stocks):
    #         stock_w, stock_h = len(stock[0]), len(stock)
    #         remaining_space[i] = stock_w * stock_h

    #     # Sort products by area (largest first) and demand
    #     sorted_products = sorted(
    #         [p for p in products if p["quantity"] > 0],
    #         key=lambda p: (p["size"][0] * p["size"][1] * p["quantity"]),
    #         reverse=True
    #     )

    #     for prod in sorted_products:
    #         if prod["quantity"] <= 0:
    #             continue

    #         prod_size = prod["size"]
    #         prod_area = prod_size[0] * prod_size[1]

    #         # Try each stock
    #         for i, stock in enumerate(stocks):
    #             stock_w, stock_h = len(stock[0]), len(stock)
                
    #             if remaining_space[i] < prod_area:
    #                 continue

    #             # Try to place the product
    #             valid_positions = self._find_valid_positions(stock, prod_size)
                
    #             for pos in valid_positions:
    #                 value = self._calculate_placement_value(
    #                     prod, stock, pos, 
    #                     remaining_space[i],
    #                     prod["quantity"],
    #                     products  # Pass products for demand calculation
    #                 )
                    
    #                 if value > best_value:
    #                     best_value = value
    #                     best_action = {
    #                         "stock_idx": i,
    #                         "size": prod_size,
    #                         "position": pos
    #                     }

    #     return best_action if best_action is not None else {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # def _find_valid_positions(self, stock, size):
    #     """Find all valid positions for a product in the stock"""
    #     stock_w, stock_h = len(stock[0]), len(stock)
    #     prod_w, prod_h = size
    #     valid_positions = []

    #     # Try normal orientation
    #     for x in range(0, stock_w - prod_w + 1, prod_w):  # Align to grid
    #         for y in range(0, stock_h - prod_h + 1, prod_h):
    #             if self._can_place_(stock, (x, y), size):
    #                 valid_positions.append((x, y))

    #     return valid_positions

    # def _calculate_placement_value(self, product, stock, position, remaining_space, remaining_demand, products):
    #     """Calculate the value of placing a product at a specific position"""
    #     stock_w, stock_h = len(stock[0]), len(stock)
    #     prod_w, prod_h = product["size"]
        
    #     # Factors to consider:
    #     # 1. Remaining demand
    #     total_demand = sum(p["quantity"] for p in products)
    #     demand_factor = remaining_demand / max(1, total_demand)
        
    #     # 2. Edge alignment (prefer placing items against edges or other items)
    #     edge_alignment = (position[0] == 0 or position[0] + prod_w == stock_w or
    #                      position[1] == 0 or position[1] + prod_h == stock_h)
        
    #     # 3. Space utilization
    #     utilization = (prod_w * prod_h) / remaining_space
        
    #     # Combine factors
    #     value = (demand_factor * 0.4 + 
    #             (1 if edge_alignment else 0) * 0.3 + 
    #             utilization * 0.3)
        
    #     return value

    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.placement_plan = []
            self.current_step = 0
            self.has_planned = False
            self.used_stock_area = 0
            self.total_used_area = 0
            self.previous_observation = None
            self.overall_utilization = 0
            pass



    def _plan_placements_(self, observation):
        """Create a complete placement plan at the start"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Reset area calculations
        self.used_stock_area = 0
        self.total_used_area = 0
        
        # Get unique product sizes
        unique_sizes = set()
        for prod in products:
            if prod["quantity"] > 0:
                unique_sizes.add(tuple(prod["size"]))
        
        # If only one product size, use different strategy
        if len(unique_sizes) == 1:
            placement_plan = self._plan_single_product_size_(observation, unique_sizes.pop())
        else:
            placement_plan = self._plan_multiple_sizes_(observation)

        # Print overall utilization for used stocks
        if self.used_stock_area > 0:
            print(f"\nOverall utilization: {self.overall_utilization:.2f}%")
        return placement_plan

    def _plan_single_product_size_(self, observation, prod_size):
        """Special strategy for single product size"""
        stocks = observation["stocks"]
        total_quantity = sum(prod["quantity"] for prod in observation["products"])
        prod_w, prod_h = prod_size
        
        # Calculate how many products can fit in each stock
        stock_capacities = []
        for i, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            # Calculate maximum products that can fit
            products_per_row = stock_w // prod_w
            rows = stock_h // prod_h
            capacity = products_per_row * rows
            
            if capacity > 0:
                utilization = (capacity * prod_w * prod_h) / (stock_w * stock_h)
                stock_capacities.append((i, capacity, utilization))
        
        # Sort stocks by utilization (highest first)
        stock_capacities.sort(key=lambda x: x[2], reverse=True)
        
        placement_plan = []
        products_left = total_quantity
        
        for stock_idx, capacity, utilization in stock_capacities:
            if products_left <= 0:
                break
                
            stock = stocks[stock_idx].copy()  
            stock_w, stock_h = self._get_stock_size_(stock)
            products_to_place = min(capacity, products_left)
            
            self.used_stock_area += stock_w * stock_h
            
            placements = self._place_products_in_stock_(
                stock_idx, stock_w, stock_h, 
                prod_w, prod_h, products_to_place
            )
            
            # Update the stock grid for each placement
            for placement in placements:
                x, y = placement["position"]
                for i in range(prod_w):
                    for j in range(prod_h):
                        stock[x+i][y+j] = stock_idx
            
            placement_plan.extend(placements)
            products_left -= products_to_place
            
            used_area = products_to_place * prod_w * prod_h
            self.total_used_area += used_area
            
            if used_area > 0:
                print(f"Stock {stock_idx} utilization: {(used_area/stock_w/stock_h)*100:.2f}%")
            
        return placement_plan

    def _place_products_in_stock_(self, stock_idx, stock_w, stock_h, prod_w, prod_h, quantity):
        """Place specified number of same-size products in a stock"""
        placements = []
        count = 0
        
        for y in range(0, stock_h - prod_h + 1, prod_h):
            for x in range(0, stock_w - prod_w + 1, prod_w):
                if count >= quantity:
                    break
                    
                placements.append({
                    "stock_idx": stock_idx,
                    "size": [prod_w, prod_h],
                    "position": (x, y)
                })
                count += 1
                
            if count >= quantity:
                break
                
        return placements

    def _plan_multiple_sizes_(self, observation):
        """Strategy for multiple product sizes"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        sorted_stocks = []
        for i, stock in enumerate(stocks):
            w, h = self._get_stock_size_(stock)
            sorted_stocks.append((i, w, h, w * h))
        sorted_stocks.sort(key=lambda x: x[3], reverse=True)

        product_instances = []
        for prod in products:
            quantity = prod["quantity"]
            size = prod["size"]
            for _ in range(quantity):
                product_instances.append((size[0], size[1], size[0] * size[1]))
        
        product_instances.sort(key=lambda x: x[2], reverse=True)

        placement_plan = []
        remaining_products = product_instances.copy()
        selected_utilizations = []  # Keep track of selected utilizations
        
        # Store all possible placements with their utilization
        potential_placements = []
        
        for stock_idx, stock_w, stock_h, _ in sorted_stocks:
            if not remaining_products:  # If no products left to place
                break
                
            stock_placements = self._plan_stock_placement_(
                stock_idx, stock_w, stock_h, remaining_products, observation)
            
            if stock_placements:
                used_area = sum(p["size"][0] * p["size"][1] for p in stock_placements)
                stock_area = stock_w * stock_h
                utilization = (used_area / stock_area) * 100
                
                print(f"Stock {stock_idx} potential utilization: {utilization:.2f}%")
                
                # If utilization > 90%, use this stock immediately
                if utilization >= 90:
                    self.used_stock_area += stock_area
                    self.total_used_area += used_area
                    selected_utilizations.append(utilization)  # Add to selected utilizations
                    print(f"Selected Stock {stock_idx} with high utilization: {utilization:.2f}%")
                    return stock_placements
                
                # Otherwise, add to potential placements
                potential_placements.append({
                    'stock_idx': stock_idx,
                    'placements': stock_placements,
                    'utilization': utilization,
                    'stock_w': stock_w,
                    'stock_h': stock_h,
                    'used_area': used_area,
                    'stock_area': stock_area
                })

        if not potential_placements:
            return []  # Return empty list if no placements possible

        # If no high utilization found, use the best available
        best_placement = max(potential_placements, key=lambda x: x['utilization'])
        stock_idx = best_placement['stock_idx']
        self.used_stock_area += best_placement['stock_area']
        self.total_used_area += best_placement['used_area']
        selected_utilizations.append(best_placement['utilization'])  # Add to selected utilizations
        print(f"Selected Stock {stock_idx} with best possible utilization: {best_placement['utilization']:.2f}%")

        # Calculate overall utilization as average of selected utilizations
        if selected_utilizations:
            self.overall_utilization = sum(selected_utilizations) / len(selected_utilizations)
        
        return best_placement['placements']


    def _plan_stock_placement_(self, stock_idx, stock_w, stock_h, products, observation):
        """Plan placements for a single stock"""
        placements = []
        stock = observation["stocks"][stock_idx].copy()  # Create a copy to track placements

        for prod_w, prod_h, _ in products:
            placed = False
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), [prod_w, prod_h]):
                        placements.append({
                            "stock_idx": stock_idx,
                            "size": [prod_w, prod_h],
                            "position": (x, y)
                        })
                        
                        # Update the stock grid
                        for i in range(prod_w):
                            for j in range(prod_h):
                                stock[x+i][y+j] = stock_idx
                        
                        placed = True
                        break
                if placed:
                    break

        return placements

    def get_action(self, observation, info):
        # Check if this is a new observation by looking at the stocks
        if (not self.has_planned or 
            len(self.placement_plan) == 0 or 
            self.current_step >= len(self.placement_plan) or
            observation["stocks"] != self.previous_observation["stocks"]):  
            
            # Reset all state variables
            self.placement_plan = []
            self.current_step = 0
            self.has_planned = False
            self.used_stock_area = 0
            self.total_used_area = 0
            
            # Create new placement plan
            self.placement_plan = self._plan_placements_(observation)
            self.has_planned = True
            self.current_step = 0
            
            # Store current observation for future comparison
            self.previous_observation = observation  

        if self.current_step < len(self.placement_plan):
            action = self.placement_plan[self.current_step]
            self.current_step += 1
            return action
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
