# Task 4 - Optimization Model using Linear Programming

from pulp import *

print("Starting Optimization Model...")

# 1️⃣ Create optimization problem
model = LpProblem("Factory_Profit_Maximization", LpMaximize)

# 2️⃣ Decision variables
product_A = LpVariable("Product_A", lowBound=0)
product_B = LpVariable("Product_B", lowBound=0)

# 3️⃣ Objective function (maximize profit)
# Profit: A = $20, B = $30
model += 20 * product_A + 30 * product_B

# 4️⃣ Constraints
# Machine hours constraint
model += 2 * product_A + 1 * product_B <= 100

# Labor hours constraint
model += 1 * product_A + 3 * product_B <= 90

# 5️⃣ Solve the problem
model.solve()

print("\nOptimization Status:", LpStatus[model.status])

# 6️⃣ Print optimal values
print("\nOptimal Production Plan")
print("Produce Product A:", value(product_A))
print("Produce Product B:", value(product_B))

# 7️⃣ Maximum profit
print("\nMaximum Profit:", value(model.objective))