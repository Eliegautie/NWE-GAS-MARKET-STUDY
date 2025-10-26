#!/usr/bin/env python
# coding: utf-8

# # **When do we pay operational costs?** 
# 
# **At position opening:**
# 
# - Injection/withdrawal costs are subtracted during the initial optimization
# 
# - Required spread = (Withdrawal price - Injection price) > (Injection cost + Withdrawal cost)
# 
# - If spread is insufficient → No position
# 
# **At unwind/closing:**
# 
# - Only use GROSS prices — no re-subtraction of costs
# 
# - PnL = (Current price - Locked price) × Volume
# 
# - The costs were already accounted for in the initial spread

# # **Economic logic:**
# 
# **Concrete example:**
# 
# - Opening: Buy Jun26 @ €30, Sell Jan27 @ €40
# 
# - Costs: €0.1 injection + €0.1 withdrawal = €0.2/MWh
# 
# - Net locked spread = (40 - 30) - 0.2 = €9.8/MWh ← Costs already deducted
# 
# - Unwind: Jun26 @ €32, Jan27 @ €39
# 
# - PnL = (32 - 30) + (40 - 39) = €3/MWh ← Gross prices only
# 
# **The costs are paid only once — at future physical execution, but for paper trading they are included in the required spread.**
