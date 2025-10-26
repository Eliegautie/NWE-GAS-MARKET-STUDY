#!/usr/bin/env python
# coding: utf-8

# # **GAS STORAGE OPTIMIZATION**

# In[1]:


import sys
sys.path.append(r"C:\Users\Elie\Elie code\Function\TTF")
import warnings
warnings.filterwarnings('ignore')
# import des fonctions
from Storage_Optimization import StorageOptimization


# **Function import**

# In[2]:


StorageOptimization = StorageOptimization()


# **Load the df with all the forward contract price (Oct 25 -> Mars 27) since 2025**

# In[3]:


TTF = StorageOptimization.get_contracts_dataframe(market='TTF', contract_type='monthly')
TTF = TTF[:-1]
TTF


# **Stores the df in the StorageOptimization for processing**

# In[4]:


StorageOptimization.load_forward_curve_data(TTF)  


# # **STEP 1: Set storage parameters first**

# In[5]:


StorageOptimization.set_storage_parameters(
    capacity_gwh=175.8,
    max_injection_gwh_per_day=5.86,
    max_withdrawal_gwh_per_day=5.86,
    fixed_cost_euros=200000
)


# # **STEP 2: Launch the Backtest**

# In[6]:


# Appeler la fonction et récupérer tous les résultats
results = StorageOptimization.calculate_rolling_intrinsic_value_with_costs()  


# **A. Current Actual Performance (From Your Backtest)**

# from simulation:
# - Rebalancing Profit (10 months): 382,365 €
# - Number of Trades: 39
# - Fixed Cost (annual): 200,000 €
# 
# Simple math:
# - Net P&L so far = 367,070 € - 200,000 €
#                  = **182,365 €**

# **C. Projection to Full Year**

# I start trading in Jan25. If we continue at same pace for 2 more months (until Dec25) :
# 
# - Profit per month = 382,365 € / (10) = 38,236 €
# - Total annual profit = 38,236 € * 12 = 458 838 €
# - Annual net P&L = 458 838 € - €200,000 € = **258 838 €**

# In[ ]:




