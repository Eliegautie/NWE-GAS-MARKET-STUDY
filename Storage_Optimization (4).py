import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from matplotlib import gridspec

# Plot style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StorageOptimization:

    def __init__(self, data_path: str = r"C:\Users\Elie\Datas\TradingView\TTF curve"):
        self.data_path = data_path
        
        self.month_codes = {
            'F': {'month': 1, 'name': 'Jan'},   'G': {'month': 2, 'name': 'Feb'},
            'H': {'month': 3, 'name': 'Mar'},   'J': {'month': 4, 'name': 'Apr'},
            'K': {'month': 5, 'name': 'May'},   'M': {'month': 6, 'name': 'Jun'},
            'N': {'month': 7, 'name': 'Jul'},   'Q': {'month': 8, 'name': 'Aug'},
            'U': {'month': 9, 'name': 'Sep'},   'V': {'month': 10, 'name': 'Oct'},
            'X': {'month': 11, 'name': 'Nov'},  'Z': {'month': 12, 'name': 'Dec'}
        }
        
        # Data cache
        self._data = None
        
        # Configuring composite contracts
        self.composite_contracts = {
            'Q425': ['Oct25', 'Nov25', 'Dec25'],
            'Q126': ['Jan26', 'Feb26', 'Mar26'],
            'Q226': ['Apr26', 'May26', 'Jun26'],
            'Q326': ['Jul26', 'Aug26', 'Sep26'],
            'Q426': ['Oct26', 'Nov26', 'Dec26'],
            'Winter25': ['Oct25', 'Nov25', 'Dec25', 'Jan26', 'Feb26', 'Mar26'],
            'Winter26': ['Oct26', 'Nov26', 'Dec26', 'Jan27', 'Feb27', 'Mar27'],
            'Summer26': ['Apr26', 'May26', 'Jun26', 'Jul26', 'Aug26', 'Sep26'],
            'Cal26': ['Jan26', 'Feb26', 'Mar26', 'Apr26', 'May26', 'Jun26',
                     'Jul26', 'Aug26', 'Sep26', 'Oct26', 'Nov26', 'Dec26']
        }

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse filename to extract contract information"""
        patterns = [
            (r'ICEENDEX_DLY_PEG([FGHJKMNQUVXZ])(\d{4})', 'PEG'),
            (r'ICEENDEX_DLY_TFM([FGHJKMNQUVXZ])(\d{4})', 'TTF'),
            (r'ICEENDEX_DLY_THE([FGHJKMNQUVXZ])(\d{4})', 'THE'),
        ]
        
        for pattern, market in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                month_code, year = match.groups()
                month_info = self.month_codes.get(month_code.upper())
                if month_info:
                    return {
                        'market': market,
                        'month': month_info['month'],
                        'month_name': month_info['name'],
                        'year': int(year),
                        'contract_id': f"{market}{month_info['name']}{str(year)[-2:]}"
                    }
        return None

    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column in DataFrame based on possible names"""
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        return None

    def _load_data(self):
        """Load all daily CSV data from specified directory"""
        if self._data is not None:
            return
            
        self._data = {}
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        print(f"Loading daily data from {self.data_path}...")
        print(f"{len(csv_files)} CSV files found")
        
        loaded_count = 0
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            contract_info = self._parse_filename(filename)
            
            if contract_info:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Search for the necessary columns
                    date_col = self._find_column(df, ['time', 'date', 'timestamp'])
                    close_col = self._find_column(df, ['close'])
                    
                    if date_col is None or close_col is None:
                        print(f"Missing columns for {filename}")
                        continue
                    
                    # Data processing
                    if df[date_col].dtype in ['int64', 'float64']:
                        df['Date'] = pd.to_datetime(df[date_col], unit='s')
                    else:
                        df['Date'] = pd.to_datetime(df[date_col])
                    
                    df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
                    df = df[df['Date'] >= '2025-01-01']
                    df = df.dropna(subset=['Close']).sort_values('Date')
                    
                    if len(df) == 0:
                        continue
                    
                    self._data[contract_info['contract_id']] = df[['Date', 'Close']].copy()
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"❌ Error with {filename}: {str(e)}")
                    continue

    def _get_monthly_expiration_date(self, month_contract: str) -> Optional[pd.Timestamp]:        
        """
        Calculates the expiration date for a monthly contract (e.g., 'Oct25')
        Rule: 2 business days before the 1st of the delivery month
        """
        month_mapping = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        if len(month_contract) < 5:
            return None
        
        month_str = month_contract[:3]
        year_str = month_contract[3:]
        
        if month_str not in month_mapping:
            return None
        
        month = month_mapping[month_str]
        year = 2000 + int(year_str)  # Convert '25' to 2025
        
        # Date of the 1st of the delivery month
        first_day = pd.Timestamp(year=year, month=month, day=1)
        
        # Go back to find the second business day before
        business_days_back = 0
        current_date = first_day
        
        while business_days_back < 2:
            current_date -= timedelta(days=1)
            # Check if it is a working day (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                business_days_back += 1
        
        return current_date

    def get_contracts_dataframe(self, market: str = 'TTF', 
                                contract_type: str = 'monthly') -> pd.DataFrame:        
        """
        Returns a DataFrame with all forward contracts in a market in columns
        """
        self._load_data()
        
        if contract_type == 'monthly':
            # Retrieve all available monthly contracts for the market
            available_contracts = [
                contract for contract in self._data.keys() 
                if contract.startswith(market)
            ]
            
            # Function to extract the sort date from a monthly contract
            def get_contract_sort_date(contract: str) -> pd.Timestamp:
                contract_name = contract[len(market):]
                expiry = self._get_monthly_expiration_date(contract_name)
                return expiry if expiry else pd.Timestamp('2099-12-31')
            
            # Sort contracts chronologically
            sorted_contracts = sorted(available_contracts, key=get_contract_sort_date)
            
        elif contract_type == 'composite':
            # Retrieve available composite contracts
            sorted_contracts = []
            
            # Function to extract the sort date from a composite
            def get_composite_sort_date(composite_name: str) -> pd.Timestamp:
                first_month = self.composite_contracts[composite_name][0]
                expiry = self._get_monthly_expiration_date(first_month)
                return expiry if expiry else pd.Timestamp('2099-12-31')
            
            # Sort composites chronologically
            sorted_composites = sorted(
                self.composite_contracts.keys(), 
                key=get_composite_sort_date
            )
            sorted_contracts = sorted_composites
            
        else:
            raise ValueError(f"contract_type must be 'monthly' or 'composite', not '{contract_type}'")
        
        # Create the DataFrame with all unique dates
        all_dates = set()
        contract_data = {}
        
        for contract in sorted_contracts:
            if contract_type == 'monthly':
                contract_id = contract
                if contract_id in self._data:
                    df = self._data[contract_id].copy()
                    all_dates.update(df['Date'].values)
                    contract_data[contract] = df.set_index('Date')['Close']
            
            elif contract_type == 'composite':
                monthly_contracts = [f"{market}{m}" for m in self.composite_contracts[contract]]
                
                # Retrieve all monthly data
                monthly_dfs = []
                for mc in monthly_contracts:
                    if mc in self._data:
                        monthly_dfs.append(self._data[mc].copy())
                
                if monthly_dfs:
                    composite_dates = set()
                    for df in monthly_dfs:
                        composite_dates.update(df['Date'].values)
                    
                    # Create a series with average prices by date
                    composite_series = []
                    for date in sorted(composite_dates):
                        prices = []
                        for df in monthly_dfs:
                            date_data = df[df['Date'] == date]
                            if not date_data.empty:
                                prices.append(date_data['Close'].iloc[0])
                        
                        if prices:
                            composite_series.append({
                                'Date': date,
                                'Close': sum(prices) / len(prices)
                            })
                    
                    if composite_series:
                        comp_df = pd.DataFrame(composite_series)
                        all_dates.update(comp_df['Date'].values)
                        contract_data[contract] = comp_df.set_index('Date')['Close']
        
        # Create the final DataFrame
        if not contract_data:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(index=sorted(all_dates))
        result_df.index.name = 'Date'
        
        # Add each contract as a column
        for contract in sorted_contracts:
            if contract in contract_data:
                result_df[contract] = contract_data[contract]
        
        # Sort by date and delete completely empty rows
        result_df = result_df.sort_index()
        result_df = result_df.dropna(how='all')
        result_df = result_df.fillna(method='ffill')
        
        return result_df

    def load_forward_curve_data(self, df):        
        """
        LOAD YOUR FORWARD CURVE DATAFRAME INTO THE ANALYZER
        """
        print("=== LOADING FORWARD CURVE DATA ===")
        
        # Store the DataFrame
        self._forward_curve = df         
        return self

    def set_storage_parameters(self, capacity_gwh=175.8, max_injection_gwh_per_day=5.86, 
                              max_withdrawal_gwh_per_day=5.86, fixed_cost_euros=400000,
                              variable_injection_cost=0.1, variable_withdrawal_cost=0.1):        
        """
        DEFINE PHYSICAL STORAGE CONSTRAINTS AND COSTS
        """
        self.storage_params = {
            'capacity_gwh': capacity_gwh,
            'max_injection_gwh_per_day': max_injection_gwh_per_day,
            'max_withdrawal_gwh_per_day': max_withdrawal_gwh_per_day,
            'fixed_cost_euros': fixed_cost_euros,
            'variable_injection_cost': variable_injection_cost,
            'variable_withdrawal_cost': variable_withdrawal_cost,
        }
        self._calculate_storage_costs()

    def _calculate_storage_costs(self):
        """
        CALCULATE COST COMPONENTS PER MWH
        """
        if not hasattr(self, 'storage_params'):
            self.set_storage_parameters()
        
        params = self.storage_params
        
        # Convert fixed annual cost to daily cost per MWh
        fixed_cost_per_mwh_per_year = params['fixed_cost_euros'] / (params['capacity_gwh'] * 1000)
        
        self.storage_costs = {
            'fixed_cost_per_mwh_per_day': fixed_cost_per_mwh_per_year / 365,
            'injection_cost_per_mwh': params['variable_injection_cost'],
            'withdrawal_cost_per_mwh': params['variable_withdrawal_cost'],
            'total_variable_cost_per_mwh': params['variable_injection_cost'] + params['variable_withdrawal_cost']
        }
        
        print("=== 💰 STORAGE COST BREAKDOWN ===")
        print(f"🏠 Fixed Cost: {self.storage_costs['fixed_cost_per_mwh_per_day']:.4f} €/MWh/day")
        print(f"⬇️  Injection Cost: {self.storage_costs['injection_cost_per_mwh']} €/MWh")   
        print(f"⬆️  Withdrawal Cost: {self.storage_costs['withdrawal_cost_per_mwh']} €/MWh")
        print(f"📊 Total Variable Cost: {self.storage_costs['total_variable_cost_per_mwh']} €/MWh")
        
    def _optimize_storage_intrinsic(self, forward_curve: Dict, capacity_mwh: float,
                                   max_inj_mwh_per_day: float, max_with_mwh_per_day: float,
                                   contracts: List[str]):
        """
        SIMPLE STORAGE OPTIMIZATION - ENSURE POSITIVE VALUE
        """
        try:
            from scipy.optimize import linprog
            
            n_contracts = len(contracts)
            
            # Objective: Maximize profit = ∑(withdrawal_price × withdrawal - injection_price × injection)
            c = []
            for contract in contracts:
                price = forward_curve.get(contract, 0)
                c.append(price)      # Injection cost
                c.append(-price)     # Withdrawal revenue
            
            # Constraints
            A_ub = []
            b_ub = []
            
            # 1. Total injection volume <= capacity
            injection_constraint = [1 if i % 2 == 0 else 0 for i in range(2 * n_contracts)]
            A_ub.append(injection_constraint)
            b_ub.append(capacity_mwh)
            
            # 2. Total withdrawal volume <= capacity  
            withdrawal_constraint = [0 if i % 2 == 0 else 1 for i in range(2 * n_contracts)]
            A_ub.append(withdrawal_constraint)
            b_ub.append(capacity_mwh)
            
            # 3. Monthly injection rate limits (30 days per month)
            max_monthly_inj = max_inj_mwh_per_day * 30
            for i in range(n_contracts):
                rate_constraint = [0] * (2 * n_contracts)
                rate_constraint[2 * i] = 1
                A_ub.append(rate_constraint)
                b_ub.append(max_monthly_inj)
            
            # 4. Monthly withdrawal rate limits
            max_monthly_with = max_with_mwh_per_day * 30
            for i in range(n_contracts):
                rate_constraint = [0] * (2 * n_contracts)
                rate_constraint[2 * i + 1] = 1
                A_ub.append(rate_constraint)
                b_ub.append(max_monthly_with)
            
            # 5. Temporal constraints: Can't withdraw more than injected so far
            for i in range(n_contracts):
                temporal_constraint = [0] * (2 * n_contracts)
                # Sum of injections up to period i
                for j in range(i + 1):
                    temporal_constraint[2 * j] = -1
                # Sum of withdrawals up to period i  
                for j in range(i + 1):
                    temporal_constraint[2 * j + 1] = 1
                A_ub.append(temporal_constraint)
                b_ub.append(0)
            
            # Equality constraint: Total injection = Total withdrawal
            A_eq = [[1 if i % 2 == 0 else -1 for i in range(2 * n_contracts)]]
            b_eq = [0]
            
            bounds = [(0, None)] * (2 * n_contracts)
            
            # Solve optimization
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if not result.success:
                return self._simple_storage_optimization(forward_curve, capacity_mwh, contracts)
            
            # Extract results
            optimal_positions = {}
            
            for i, contract in enumerate(contracts):
                injection = result.x[2 * i]
                withdrawal = result.x[2 * i + 1]
                
                optimal_positions[contract] = {
                    'injection': injection,
                    'withdrawal': withdrawal,
                    'net': injection - withdrawal
                }
            
            intrinsic_value = -result.fun  # Convert back to maximization
            
            if intrinsic_value < 0:
                intrinsic_value = 0
                optimal_positions = {}
                for contract in contracts:
                    optimal_positions[contract] = {
                        'injection': 0,
                        'withdrawal': 0,
                        'net': 0
                    }
            
            return optimal_positions, intrinsic_value, {}
            
        except ImportError:
            print("scipy not available, using simple optimization")
            return self._simple_storage_optimization(forward_curve, capacity_mwh, contracts)

    def _simple_storage_optimization(self, forward_curve: Dict, capacity_mwh: float, contracts: List[str]):
        """
        SIMPLE OPTIMIZATION WHEN SCIPY IS NOT AVAILABLE
        Basic strategy: Buy cheapest 3 months, sell most expensive 3 months
        """
        valid_prices = {k: v for k, v in forward_curve.items() if not pd.isna(v)}
        
        if len(valid_prices) < 2:
            return {}, 0, {}
        
        sorted_contracts = sorted(valid_prices.keys(), key=lambda x: valid_prices[x])
        
        optimal_positions = {}
        
        # Inject in cheapest 3 months
        injection_contracts = sorted_contracts[:3]
        for contract in injection_contracts:
            volume = capacity_mwh / len(injection_contracts)
            optimal_positions[contract] = {
                'injection': volume,
                'withdrawal': 0,
                'net': volume
            }
        
        # Withdraw in most expensive 3 months  
        withdrawal_contracts = sorted_contracts[-3:]
        for contract in withdrawal_contracts:
            volume = capacity_mwh / len(withdrawal_contracts)
            if contract in optimal_positions:
                optimal_positions[contract]['withdrawal'] = volume
                optimal_positions[contract]['net'] = optimal_positions[contract]['injection'] - volume
            else:
                optimal_positions[contract] = {
                    'injection': 0,
                    'withdrawal': volume,
                    'net': -volume
                }
        
        # Calculate intrinsic value
        intrinsic_value = 0
        for contract, position in optimal_positions.items():
            price = valid_prices[contract]
            intrinsic_value += position['withdrawal'] * price - position['injection'] * price
        
        return optimal_positions, intrinsic_value, {}

    def _optimize_storage_intrinsic_with_costs(self, forward_curve: Dict, capacity_mwh: float,
                                             max_inj_mwh_per_day: float, max_with_mwh_per_day: float,
                                             contracts: List[str], injection_cost: float, withdrawal_cost: float):
        """
        OPTIMIZATION WITH COSTS - CORRECT VERSION
        """
        # Use gross prices in optimization
        optimal_positions, intrinsic_value, _ = self._optimize_storage_intrinsic(
            forward_curve, capacity_mwh, max_inj_mwh_per_day, max_with_mwh_per_day, contracts
        )
        
        # Explicitly subtract costs
        total_injection_cost = 0
        total_withdrawal_cost = 0
        
        for contract, position in optimal_positions.items():
            injection_volume = position.get('injection', 0)
            withdrawal_volume = position.get('withdrawal', 0)
            
            total_injection_cost += injection_volume * injection_cost
            total_withdrawal_cost += withdrawal_volume * withdrawal_cost
        
        # Adjust the intrinsic value
        adjusted_intrinsic = intrinsic_value - total_injection_cost - total_withdrawal_cost
        
        # ENSURE THAT THE FINAL VALUE IS ≥ 0
        if adjusted_intrinsic < 0:
            adjusted_intrinsic = 0
            # Close positions if unprofitable
            optimal_positions = {}
            for contract in contracts:
                optimal_positions[contract] = {
                    'injection': 0,
                    'withdrawal': 0,
                    'net': 0
                }
        
        return optimal_positions, adjusted_intrinsic, {}

    def _calculate_realized_pnl_with_costs(self, current_positions, locked_prices, current_curve, 
                                         injection_cost, withdrawal_cost):
        """
        CORRECT realized PnL calculation - WITHOUT re-subtracting costs
        """
        realized_pnl = 0
        actual_sales = []
        
        for contract, old_pos in current_positions.items():
            if contract in current_curve and contract in locked_prices:
                current_price = current_curve[contract]
                
                old_inj = old_pos.get('injection', 0)
                old_with = old_pos.get('withdrawal', 0)
                
                # Sell injection positions - GROSS prices
                if old_inj > 0:
                    locked_price_inj = locked_prices[contract]['injection']
                    # DO NOT subtract injection_cost here - already accounted for at opening
                    sale_pnl = (current_price - locked_price_inj) * old_inj
                    realized_pnl += sale_pnl
                    actual_sales.append(f"SOLD {old_inj:,.0f} MWh of {contract} (injection) for {sale_pnl:,.0f}€ PnL")
                
                # Buy back withdrawal positions - GROSS prices  
                if old_with > 0:
                    locked_price_with = locked_prices[contract]['withdrawal']
                    # DO NOT subtract withdrawal_cost here - already accounted for at opening
                    sale_pnl = (locked_price_with - current_price) * old_with
                    realized_pnl += sale_pnl
                    actual_sales.append(f"BOUGHT {old_with:,.0f} MWh of {contract} (withdrawal) for {sale_pnl:,.0f}€ PnL")
        
        return realized_pnl, actual_sales

    def should_rebalance(self, new_intrinsic, base_intrinsic, realized_pnl, min_threshold=1000):
        """
        Check if rebalancing is truly profitable
        Simplified version - only check net benefit
        """
        net_benefit = (new_intrinsic + realized_pnl) - base_intrinsic
        return net_benefit > min_threshold

    def calculate_rolling_intrinsic_value_with_costs(self):        
        """
        CORRECT VERSION - Lock spreads and rebalance only if profitable
        """
        print("\n=== ROLLING INTRINSIC VALUE - LOCKED SPREADS ===")
        
        if self._forward_curve is None:
            print("❌ No forward curve data loaded")
            return None, None, None, None
        
        # Storage parameters
        capacity_mwh = self.storage_params['capacity_gwh'] * 1000
        max_inj_mwh_per_day = self.storage_params['max_injection_gwh_per_day'] * 1000
        max_with_mwh_per_day = self.storage_params['max_withdrawal_gwh_per_day'] * 1000
        
        # Get costs
        injection_cost = self.storage_costs['injection_cost_per_mwh']
        withdrawal_cost = self.storage_costs['withdrawal_cost_per_mwh']
        fixed_cost = self.storage_params.get('fixed_cost_euros', 200000)
        
        print(f"💰 COSTS: Injection={injection_cost}€/MWh, Withdrawal={withdrawal_cost}€/MWh, Fixed={fixed_cost:,.0f}€")
        print(f"📊 Capacity: {capacity_mwh:,.0f} MWh")
        
        # Get all available dates
        all_dates = self._forward_curve.index
        print(f"📅 Available dates: {len(all_dates)} days")
        
        # Initialize
        current_positions = {}
        locked_prices = {}
        total_pnl = 0
        positions_history = []
        rebalancing_summary = []
        
        # Start with initial intrinsic value
        first_date = all_dates[0]
        first_curve = self._forward_curve.loc[first_date].dropna().to_dict()
        
        print(f"📈 First day: {first_date}, available contracts: {len(first_curve)}")
        
        current_positions, initial_intrinsic, _ = self._optimize_storage_intrinsic_with_costs(
            first_curve, capacity_mwh, max_inj_mwh_per_day, max_with_mwh_per_day, 
            list(first_curve.keys()), injection_cost, withdrawal_cost
        )
        
        print(f"🎯 Initial intrinsic value: {initial_intrinsic:,.0f} €")
        
        # Initialize locked prices
        for contract in first_curve:
            locked_prices[contract] = {
                'injection': first_curve[contract],
                'withdrawal': first_curve[contract]
            }
        
        base_intrinsic = initial_intrinsic
        
        # Store initial position
        positions_history.append({
            'date': first_date,
            'positions': current_positions.copy(),
            'intrinsic_value': initial_intrinsic,
            'economic_value': initial_intrinsic,
            'total_pnl': 0,
            'realized_pnl': 0,
            'net_benefit': 0
        })
        
        rebalance_count = 0
        
        # Check each day for rebalancing opportunities
        for i, current_date in enumerate(all_dates[1:], 1):
            current_curve = self._forward_curve.loc[current_date].dropna().to_dict()
            
            if not current_curve:
                continue
            
            # Calculate new optimal positions
            new_positions, new_intrinsic, _ = self._optimize_storage_intrinsic_with_costs(
                current_curve, capacity_mwh, max_inj_mwh_per_day, max_with_mwh_per_day,
                list(current_curve.keys()), injection_cost, withdrawal_cost
            )
            
            # ECONOMIC VALUE BASED ON LOCKED PRICES ONLY
            current_economic_value = 0
            total_injection_volume = 0
            total_withdrawal_volume = 0
            
            for contract, position in current_positions.items():
                if contract in locked_prices:
                    injection_volume = position.get('injection', 0)
                    withdrawal_volume = position.get('withdrawal', 0)
                    
                    total_injection_volume += injection_volume
                    total_withdrawal_volume += withdrawal_volume
                    
                    # Injection: locked cost (negative)
                    if injection_volume > 0 and locked_prices[contract]['injection'] is not None:
                        locked_price_inj = locked_prices[contract]['injection']
                        current_economic_value -= injection_volume * locked_price_inj
                    
                    # Withdrawal: locked revenue (positive)  
                    if withdrawal_volume > 0 and locked_prices[contract]['withdrawal'] is not None:
                        locked_price_with = locked_prices[contract]['withdrawal']
                        current_economic_value += withdrawal_volume * locked_price_with
            
            # Subtract operational costs
            operational_costs = (total_injection_volume * injection_cost + 
                               total_withdrawal_volume * withdrawal_cost)
            current_economic_value -= operational_costs
            
            # Step 1: CLOSE existing positions WITH CORRECT COST CALCULATION
            realized_pnl, actual_sales = self._calculate_realized_pnl_with_costs(
                current_positions, locked_prices, current_curve, 
                injection_cost, withdrawal_cost
            )
            
            # NET BENEFIT calculation
            net_benefit = (new_intrinsic + realized_pnl) - base_intrinsic

            # DAILY DATA STORAGE
            daily_data = {
                'date': current_date,
                'economic_value': current_economic_value,
                'intrinsic_value': new_intrinsic,
                'positions': current_positions.copy(),
                'total_pnl': total_pnl,
                'realized_pnl': realized_pnl,
                'net_benefit': net_benefit
            }
            
            # Rebalance ONLY if truly profitable
            min_threshold = 1000  # € minimum to avoid micro-rebalancings
            
            should_rebalance = self.should_rebalance(new_intrinsic, base_intrinsic, realized_pnl, min_threshold)
            
            if should_rebalance:
                rebalance_count += 1
                
                # Step 2: OPEN new positions
                new_openings = []
                for contract, new_pos in new_positions.items():
                    new_inj = new_pos.get('injection', 0)
                    new_with = new_pos.get('withdrawal', 0)
                    
                    if new_inj > 0:
                        new_openings.append(f"BOUGHT {new_inj:,.0f} MWh of {contract} (injection)")
                    if new_with > 0:
                        new_openings.append(f"SOLD {new_with:,.0f} MWh of {contract} (withdrawal)")
                
                # Store rebalancing details
                rebalancing_summary.append({
                    'date': current_date,
                    'old_intrinsic': base_intrinsic,
                    'new_intrinsic': new_intrinsic,
                    'realized_pnl': realized_pnl,  
                    'net_benefit': net_benefit,
                    'actual_sales': actual_sales,    
                    'new_openings': new_openings      
                })
                
                # Execute rebalance
                total_pnl += net_benefit
                current_positions = new_positions
                
                # Update locked prices with NEW positions
                locked_prices = {}
                for contract in current_curve:
                    new_inj = new_positions.get(contract, {}).get('injection', 0)
                    new_with = new_positions.get(contract, {}).get('withdrawal', 0)
                    
                    locked_prices[contract] = {
                        'injection': current_curve[contract] if new_inj > 0 else None,
                        'withdrawal': current_curve[contract] if new_with > 0 else None
                    }
                
                base_intrinsic = new_intrinsic
            
            # STORE EVERY DAY
            positions_history.append(daily_data)
        
        # Final results
        final_value = base_intrinsic

        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   • Total days: {len(all_dates)}")
        print(f"   • Rebalances executed: {rebalance_count}")
        print(f"   • Total PnL: {total_pnl:,.0f} €")
        print(f"   • Final value: {final_value:,.0f} €")

        # Risk metrics
        risk_metrics = self.calculate_simple_risk_metrics(positions_history)
        
        # Display summary table
        self._display_summary_table(rebalancing_summary, positions_history, final_value, total_pnl, fixed_cost)

        print(f"\n🔴 RISK METRICS (Storage Intrinsic Strategy):")
        print(f"    • 1-day VaR (95%): €{risk_metrics['var_95_absolute']:,.0f}")
        print(f"    • Volatility on Cash Days: €{risk_metrics['daily_volatility_€']:,.0f}")
        print(f"    • Max Daily Cash Loss: €{risk_metrics['max_daily_loss_€']:,.0f}")
        print(f"    • Zero-Risk Days: {risk_metrics['zero_risk_days_ratio']:.1%}")
        print(f"    💡 Note: Locked positions have ZERO price risk - metrics based on CASH movements only")
        
        return final_value, total_pnl, positions_history, rebalancing_summary

    def calculate_simple_risk_metrics(self, positions_history):
        """
        Risk metrics based on REAL CASH only - CORRECT VERSION
        In rolling intrinsic strategy, locked positions have ZERO price risk
        Only cash movements from rebalancing create risk
        """
        # Calculate accumulated cash from REALIZED PnL only
        accumulated_cash = 0
        daily_cash_values = []
        
        for position in positions_history:
            # Only add realized PnL to accumulated cash
            if 'realized_pnl' in position:
                accumulated_cash += position['realized_pnl']
            daily_cash_values.append(accumulated_cash)
        
        # Calculate CASH changes only
        daily_cash_changes = []
        for i in range(1, len(daily_cash_values)):
            change = daily_cash_values[i] - daily_cash_values[i-1]  # € change in CASH
            daily_cash_changes.append(change)
        
        import numpy as np
        
        # Filter only days with cash movement (rebalancing days)
        cash_movement_days = [change for change in daily_cash_changes if abs(change) > 1]
        
        if not cash_movement_days:
            return {
                'var_95_absolute': 0,
                'daily_volatility_€': 0,
                'max_daily_loss_€': 0,
                'avg_daily_change_€': 0,
                'zero_risk_days_ratio': 1.0,
                'total_cash': accumulated_cash
            }
        
        # 1. VaR 95% - based on CASH movements only
        var_95 = np.percentile(cash_movement_days, 5)
        
        # 2. Volatility - only on days with cash movement
        daily_volatility = np.std(cash_movement_days)
        
        # 3. Maximum Daily Loss
        max_daily_loss = min(cash_movement_days)
        
        return {
            'var_95_absolute': abs(var_95),
            'daily_volatility_€': daily_volatility,
            'max_daily_loss_€': abs(max_daily_loss),
            'avg_daily_change_€': np.mean(cash_movement_days),
            'zero_risk_days_ratio': (len(daily_cash_changes) - len(cash_movement_days)) / len(daily_cash_changes),
            'total_cash': accumulated_cash
        }

    def _display_summary_table(self, rebalancing_summary: List, positions_history: List, 
                             final_value: float, total_pnl: float, fixed_cost: float):        
        """
        SIMPLIFIED DISPLAY
        """
        print(f"\n{'='*80}")
        print("📊 ROLLING INTRINSIC STRATEGY - SUMMARY")
        print(f"{'='*80}")
        
        initial_value = positions_history[0]['intrinsic_value']
        
        # Calculation of total net profit
        total_net_profit = total_pnl - fixed_cost
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"    • Initial Value: {initial_value:,.0f} €")
        print(f"    • Final Value: {final_value:,.0f} €")
        print(f"    • Trading PnL: {total_pnl:,.0f} €")
        print(f"    • Fixed Cost: -{fixed_cost:,.0f} €")
        print(f"    • TOTAL NET PROFIT: {total_net_profit:,.0f} €")
        
        if rebalancing_summary:
            print(f"\n📈 REBALANCINGS ({len(rebalancing_summary)} total):")
            print(f"{'Date':<12} {'Old Value':<12} {'New Value':<12} {'Realized PnL':<12} {'Net Benefit':<12}")
            print(f"{'-'*70}")
            
            # Show ALL rebalances
            for rebalance in rebalancing_summary:
                print(f"{rebalance['date'].strftime('%Y-%m-%d'):<12} "
                      f"{rebalance['old_intrinsic']:>11,.0f}€ "
                      f"{rebalance['new_intrinsic']:>11,.0f}€ "
                      f"{rebalance['realized_pnl']:>11,.0f}€ "
                      f"{rebalance['net_benefit']:>11,.0f}€")

    # Other utility methods...
    def _get_main_trades_clean(self, old_positions: Dict, new_positions: Dict) -> str:        
        """Get description of significant trades - CLEAN VERSION"""
        trades = []
        all_contracts = set(list(old_positions.keys()) + list(new_positions.keys()))
        
        for contract in all_contracts:
            old_inj = old_positions.get(contract, {}).get('injection', 0)
            new_inj = new_positions.get(contract, {}).get('injection', 0)
            old_with = old_positions.get(contract, {}).get('withdrawal', 0)
            new_with = new_positions.get(contract, {}).get('withdrawal', 0)
            
            # New injections
            if new_inj > old_inj:
                trades.append(f"Inject {new_inj - old_inj:,.0f} in {contract}")
            
            # New withdrawals  
            elif new_with > old_with:
                trades.append(f"Withdraw {new_with - old_with:,.0f} from {contract}")
            
            # Closed injections
            elif old_inj > new_inj:
                trades.append(f"Close injection {old_inj - new_inj:,.0f} in {contract}")
            
            # Closed withdrawals
            elif old_with > new_with:
                trades.append(f"Close withdrawal {old_with - new_with:,.0f} in {contract}")
        
        return ", ".join(trades[:4]) if trades else "Portfolio adjustment"