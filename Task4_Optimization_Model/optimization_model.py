# Task 4 - Optimization Model using Linear Programming

from pulp import *

print("Starting Optimization Model...")

model = LpProblem("Factory_Profit_Maximization", LpMaximize)

product_A = LpVariable("Product_A", lowBound=0)
product_B = LpVariable("Product_B", lowBound=0)

model += 20 * product_A + 30 * product_B

model += 2 * product_A + 1 * product_B <= 100

model += 1 * product_A + 3 * product_B <= 90

model.solve()

print("\nOptimization Status:", LpStatus[model.status])

print("\nOptimal Production Plan")
print("Produce Product A:", value(product_A))
print("Produce Product B:", value(product_B))

print("\nMaximum Profit:", value(model.objective))