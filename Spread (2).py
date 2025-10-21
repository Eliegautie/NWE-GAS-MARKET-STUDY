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

class SpreadAnalyzer:
    """
    Spread analyzer for different energy markets (TTF, PEG, THE, etc.)
    """
            
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
        
        # Composite contracts configuration
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
        """Find a column in DataFrame based on a list of possible names"""
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
        
        
        loaded_count = 0
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            contract_info = self._parse_filename(filename)
            
            if contract_info:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Search for necessary columns
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
                    
                    latest_price = df['Close'].iloc[-1]
                    latest_date = df['Date'].iloc[-1]
                    
                    
                except Exception as e:
                    print(f"❌ Error with {filename}: {str(e)}")
                    continue
        
        print(f"\n🎯 {loaded_count} daily contracts successfully loaded\n")

    def _get_contract_expiration_date(self, contract: str, contract_type: str) -> Optional[pd.Timestamp]:
        """
        Calculate contract expiration date (2 business days before the 1st of delivery month)
        """
        if contract_type == 'composite' and contract in self.composite_contracts:
            # For composites, take the first month of the list
            first_month = self.composite_contracts[contract][0]
            return self._get_monthly_expiration_date(first_month)
        else:
            # For monthly contracts
            return self._get_monthly_expiration_date(contract)
    
    def _get_monthly_expiration_date(self, month_contract: str) -> Optional[pd.Timestamp]:
        """
        Calculate expiration date for a monthly contract (ex: 'Oct25')
        Rule: 2 business days before the 1st of the month
        """
        # Parse contract name (ex: 'Oct25' -> October 2025)
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
        
        # Date of the 1st of delivery month
        first_day = pd.Timestamp(year=year, month=month, day=1)
        
        # Go back to find the 2nd business day before
        business_days_back = 0
        current_date = first_day
        
        while business_days_back < 2:
            current_date -= timedelta(days=1)
            # Check if it's a business day (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                business_days_back += 1
        
        return current_date
    
    def _is_contract_expired(self, contract: str, contract_type: str, reference_date: pd.Timestamp) -> bool:
        """
        Check if a contract is expired at a given date
        """
        expiration_date = self._get_contract_expiration_date(contract, contract_type)
        if expiration_date is None:
            return False
        
        return reference_date > expiration_date
    
    def _get_contract_price_with_expiration(self, market: str, contract: str, 
                                          contract_type: str, reference_date: Optional[pd.Timestamp] = None) -> Optional[Dict]:
        """
        Get contract price considering expiration
        If expired, returns price frozen at last trading day
        """
        if reference_date is None:
            reference_date = pd.Timestamp.today().normalize()
        
        expiration_date = self._get_contract_expiration_date(contract, contract_type)
        
        # If contract expired, take price at expiration day
        if expiration_date and reference_date > expiration_date:
            target_date = expiration_date
            is_expired = True
        else:
            target_date = reference_date
            is_expired = False
        
        # Get price at target date
        if contract_type == 'composite' and contract in self.composite_contracts:
            monthly_contracts = [f"{market}{m}" for m in self.composite_contracts[contract]]
            composite_data = self._calculate_composite_price(monthly_contracts)
        else:
            contract_id = f"{market}{contract}"
            price_data = self._get_latest_close_price(contract_id)
        
        if price_data:
            price_data['is_expired'] = is_expired
            price_data['expiration_date'] = expiration_date
        
        return price_data
    
    def _get_price_at_specific_date(self, contract_id: str, target_date: pd.Timestamp) -> Optional[Dict]:
        """
        Get contract price at specific date (or closest before)
        """
        if contract_id not in self._data:
            return None
        
        df = self._data[contract_id]
        df_filtered = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
        
        if df_filtered.empty:
            return None
        
        return {
            'price': df_filtered['Close'].iloc[0],
            'date': df_filtered['Date'].iloc[0],
            'contract': contract_id
        }
    
    def _calculate_composite_price_at_specific_date(self, contracts: List[str], 
                                                    target_date: pd.Timestamp) -> Optional[Dict]:
        """
        Calculate composite price at specific date considering expirations
        """
        available_contracts = [c for c in contracts if c in self._data]
        if not available_contracts:
            return None
        
        prices = []
        actual_date = None
        individual_prices = {}
        
        for contract in available_contracts:
            df = self._data[contract]
            df_filtered = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
            
            if not df_filtered.empty:
                price = df_filtered['Close'].iloc[0]
                date = df_filtered['Date'].iloc[0]
                
                prices.append(price)
                individual_prices[contract] = price
                
                if actual_date is None or date > actual_date:
                    actual_date = date
        
        if not prices:
            return None
        
        return {
            'price': np.mean(prices),
            'date': actual_date,
            'contracts': available_contracts,
            'individual_prices': individual_prices
        }
    
    def _get_latest_close_price_with_expiration(self, contract_id: str) -> Optional[Dict]:
        """
        Improved version of _get_latest_close_price that handles expiration
        """
        if contract_id not in self._data:
            return None
        
        df = self._data[contract_id]
        if df.empty:
            return None
        
        today = pd.Timestamp.today().normalize()
        
        # Extract contract name (ex: 'TTFOct25' -> 'Oct25')
        market = None
        contract_name = None
        for mkt in ['TTF', 'PEG', 'THE']:
            if contract_id.startswith(mkt):
                market = mkt
                contract_name = contract_id[len(mkt):]
                break
        
        if contract_name is None:
            # Fallback to old method if parsing fails
            df_filtered = df[df['Date'] < today]
            if df_filtered.empty:
                return None
            return {
                'price': df_filtered['Close'].iloc[-1],
                'date': df_filtered['Date'].iloc[-1],
                'contract': contract_id,
                'is_expired': False
            }
        
        # Check expiration
        expiration_date = self._get_monthly_expiration_date(contract_name)
        
        if expiration_date and today > expiration_date:
            # Expired contract: take price at expiration day
            df_filtered = df[df['Date'] <= expiration_date].sort_values('Date', ascending=False)
            is_expired = True
        else:
            # Active contract: take last available price
            df_filtered = df[df['Date'] < today]
            is_expired = False
        
        if df_filtered.empty:
            return None
        
        return {
            'price': df_filtered['Close'].iloc[0],
            'date': df_filtered['Date'].iloc[0],
            'contract': contract_id,
            'is_expired': is_expired,
            'expiration_date': expiration_date
        }
    
    def _calculate_composite_price_with_expiration(self, contracts: List[str]) -> Optional[Dict]:
        """
        Improved version of _calculate_composite_price that handles expiration
        """
        available_contracts = [c for c in contracts if c in self._data]
        if not available_contracts:
            return None
        
        today = pd.Timestamp.today().normalize()
        
        # Determine if composite is expired by looking at first contract
        first_contract = available_contracts[0]
        market = None
        contract_name = None
        for mkt in ['TTF', 'PEG', 'THE']:
            if first_contract.startswith(mkt):
                market = mkt
                contract_name = first_contract[len(mkt):]
                break
        
        # Find composite expiration date (based on first month)
        composite_expiration = self._get_monthly_expiration_date(contract_name) if contract_name else None
        is_expired = composite_expiration and today > composite_expiration
        
        # Determine reference date
        if is_expired:
            reference_date = composite_expiration
        else:
            reference_date = today
        
        latest_prices = []
        latest_date = None
        individual_prices = {}
        
        for contract in available_contracts:
            # If composite expired, take price at expiration day
            if is_expired:
                df = self._data[contract]
                df_filtered = df[df['Date'] <= composite_expiration].sort_values('Date', ascending=False)
                if not df_filtered.empty:
                    price = df_filtered['Close'].iloc[0]
                    date = df_filtered['Date'].iloc[0]
                    latest_prices.append(price)
                    individual_prices[contract] = price
                    if latest_date is None or date > latest_date:
                        latest_date = date
            else:
                # Active contract: take last available price
                df = self._data[contract]
                df_filtered = df[df['Date'] < today]
                if not df_filtered.empty:
                    price = df_filtered['Close'].iloc[-1]
                    date = df_filtered['Date'].iloc[-1]
                    latest_prices.append(price)
                    individual_prices[contract] = price
                    if latest_date is None or date > latest_date:
                        latest_date = date
        
        if not latest_prices:
            return None
        
        return {
            'price': np.mean(latest_prices),
            'date': latest_date,
            'contracts': available_contracts,
            'individual_prices': individual_prices,
            'is_expired': is_expired,
            'expiration_date': composite_expiration
        }

    def _find_target_date_for_period(self, contracts: List[str], period_name: str) -> Optional[pd.Timestamp]:
        """
        Find target date for given period respecting business days
        """
        available_contracts = [c for c in contracts if c in self._data]
        if not available_contracts:
            return None
        
        today = pd.Timestamp.today().normalize()
        
        # Find last available date (current date)
        current_date = None
        for contract in available_contracts:
            df = self._data[contract]
            df_filtered = df[df['Date'] < today]
            if not df_filtered.empty:
                contract_last_date = df_filtered['Date'].iloc[-1]
                if current_date is None or contract_last_date > current_date:
                    current_date = contract_last_date
        
        if current_date is None:
            return None
        
        # For each period, calculate target date
        if period_name == 'daily':
            # For daily: find the day before last business day
            target_date = None
            for contract in available_contracts:
                df = self._data[contract]
                df_filtered = df[df['Date'] < current_date].sort_values('Date', ascending=False)
                if len(df_filtered) >= 1:
                    target_date = df_filtered['Date'].iloc[0]
                    break
        
        elif period_name == 'weekly':
            # For weekly: go back approximately 7 calendar days to find same weekday
            target_calendar_date = current_date - timedelta(days=7)
            # Find closest business day to this date
            target_date = None
            min_diff = float('inf')
            for contract in available_contracts:
                df = self._data[contract]
                df_filtered = df[df['Date'] <= target_calendar_date].sort_values('Date', ascending=False)
                if not df_filtered.empty:
                    candidate_date = df_filtered['Date'].iloc[0]
                    diff = abs((candidate_date - target_calendar_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        target_date = candidate_date
        
        elif period_name == 'monthly':
            # For monthly: go back approximately 30 calendar days
            target_calendar_date = current_date - timedelta(days=28)
            # Find closest business day to this date
            target_date = None
            min_diff = float('inf')
            for contract in available_contracts:
                df = self._data[contract]
                df_filtered = df[df['Date'] <= target_calendar_date].sort_values('Date', ascending=False)
                if not df_filtered.empty:
                    candidate_date = df_filtered['Date'].iloc[0]
                    diff = abs((candidate_date - target_calendar_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        target_date = candidate_date
        
        return target_date

    def _calculate_composite_price_at_date(self, contracts: List[str], period_name: str) -> Optional[Dict]:
        """
        Calculate composite price at previous date according to period
        """
        available_contracts = [c for c in contracts if c in self._data]
        if not available_contracts:
            return None
        
        target_date = self._find_target_date_for_period(contracts, period_name)
        
        if target_date is None:
            return None
        
        # Calculate composite price at this date
        prices = []
        actual_date = None
        individual_prices = {}
        used_contracts = []
        
        for contract in available_contracts:
            df = self._data[contract]
            
            # Find price closest to target_date (backwards)
            df_filtered = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
            
            if not df_filtered.empty:
                price = df_filtered['Close'].iloc[0]
                date = df_filtered['Date'].iloc[0]
                
                prices.append(price)
                individual_prices[contract] = price
                used_contracts.append(contract)
                
                if actual_date is None or date > actual_date:
                    actual_date = date
        
        if not prices:
            return None
        
        return {
            'price': np.mean(prices),
            'date': actual_date,
            'contracts': used_contracts,
            'individual_prices': individual_prices
        }

    def _get_variation_color(self, variation_pct: float) -> str:
        """
        Return colored circle according to variation intensity
        """
        abs_variation = abs(variation_pct)
        
        if variation_pct > 0:
            # Increase - green gradient
            if abs_variation >= 5:
                return "🟢"  # Strong increase
            elif abs_variation >= 2:
                return "🟡"  # Moderate increase
            else:
                return "🟠"  # Slight increase
        elif variation_pct < 0:
            # Decrease - red gradient
            if abs_variation >= 5:
                return "🔴"  # Strong decrease
            elif abs_variation >= 2:
                return "🟤"  # Moderate decrease
            else:
                return "🟣"  # Slight decrease
        else:
            return "⚪"  # Stable

    def composite_variation(self, markets: Union[str, List[str], None] = None, 
                           periods: Union[str, List[str], None] = None,
                           contracts: Union[str, List[str], None] = None,
                           display: bool = True):
        """
        Calculate daily, weekly and monthly variations of composite contracts
        with expiration management - only displays final price for expired contracts
        """
        self._load_data()
        
        # Markets parameter normalization
        if markets is None:
            markets = ['TTF', 'PEG']
        elif isinstance(markets, str):
            markets = [markets.upper()]
        else:
            markets = [m.upper() for m in markets]
        
        # Periods parameter normalization
        available_periods = ['daily', 'weekly', 'monthly']
        
        if periods is None:
            periods = available_periods
        elif isinstance(periods, str):
            if periods.lower() in available_periods:
                periods = [periods.lower()]
            else:
                print(f"⚠️ Period '{periods}' not recognized. Using default periods.")
                periods = available_periods
        else:
            # Filter valid periods
            valid_periods = [p.lower() for p in periods if p.lower() in available_periods]
            if valid_periods:
                periods = valid_periods
            else:
                print("⚠️ No valid period specified. Using default periods.")
                periods = available_periods
        
        # Contracts parameter normalization
        if contracts is None:
            # Use all composite contracts by default
            contracts_to_analyze = self.composite_contracts
        else:
            if isinstance(contracts, str):
                contracts = [contracts]
            
            # Filter valid contracts
            contracts_to_analyze = {}
            for contract in contracts:
                if contract in self.composite_contracts:
                    contracts_to_analyze[contract] = self.composite_contracts[contract]
                else:
                    print(f"⚠️ Composite contract '{contract}' not found. Available contracts: {list(self.composite_contracts.keys())}")
        
        if not contracts_to_analyze:
            print("❌ No valid composite contract to analyze.")
            return {}
        
        variations_results = {}
        
        for period_name in periods:
            variations_results[period_name] = {}
            
            for market in markets:
                variations_results[period_name][market] = {}
                
                for comp_name, months in contracts_to_analyze.items():
                    # Build list of monthly contracts for this market
                    monthly_list = [f"{market}{m}" for m in months]
                    
                    # Calculate current composite price WITH expiration management
                    current_composite = self._calculate_composite_price_with_expiration(monthly_list)
                    if not current_composite:
                        continue
                    
                    # If contract expired, only display final price without variation
                    if current_composite.get('is_expired', False):
                        variations_results[period_name][market][comp_name] = {
                            'current_price': current_composite['price'],
                            'previous_price': current_composite['price'],  # Same price to avoid variations
                            'variation_abs': 0.0,
                            'variation_pct': 0.0,
                            'current_date': current_composite['date'],
                            'previous_date': current_composite['date'],  # Same date
                            'contracts_coverage': f"{len(current_composite['contracts'])}/{len(monthly_list)}",
                            'is_expired': True,
                            'expiration_date': current_composite.get('expiration_date')
                        }
                    else:
                        # Active contract: calculate normal variation
                        previous_composite = self._calculate_composite_price_at_date(
                            monthly_list, period_name
                        )
                        
                        if not previous_composite:
                            continue
                        
                        # Variation calculation
                        current_price = current_composite['price']
                        previous_price = previous_composite['price']
                        
                        variation_abs = current_price - previous_price
                        variation_pct = (variation_abs / previous_price) * 100 if previous_price != 0 else 0
                        
                        variations_results[period_name][market][comp_name] = {
                            'current_price': current_price,
                            'previous_price': previous_price,
                            'variation_abs': variation_abs,
                            'variation_pct': variation_pct,
                            'current_date': current_composite['date'],
                            'previous_date': previous_composite['date'],
                            'contracts_coverage': f"{len(current_composite['contracts'])}/{len(monthly_list)}",
                            'is_expired': False
                        }
        
        if display:
            self._display_composite_variations(variations_results)
        
        

    def _display_composite_variations(self, variations_results: Dict):
        """Display composite contract variations in formatted way"""
        
        period_names = {
            'daily': 'DAILY VARIATIONS',
            'weekly': 'WEEKLY VARIATIONS', 
            'monthly': 'MONTHLY VARIATIONS'
        }
        
        for period, period_display in period_names.items():
            if period not in variations_results:
                continue
                
            print("\n" + "=" * 100)
            print(f"{period_display}".center(100))
            print("=" * 100)
            
            # Modified table header for more clarity
            print(f"{'Contract':<12} {'Market':<8} {'Price (€/MWh)':<12} {'Var. €':<8} {'Var. %':<8} {'Status':<10} {'Date':<12} {'Cov.':<6}")
            print("-" * 100)
            
            # Collect all variations for this period for sorting
            all_contracts = []
            for market, contracts_data in variations_results[period].items():
                if not contracts_data:
                    continue
                for comp_name, data in contracts_data.items():
                    all_contracts.append({
                        'market': market,
                        'contract': comp_name,
                        'data': data,
                        'is_expired': data.get('is_expired', False)
                    })
            
            # Sort: expired contracts first, then by name
            all_contracts.sort(key=lambda x: (not x['is_expired'], x['contract']))
            
            for item in all_contracts:
                data = item['data']
                market = item['market']
                comp_name = item['contract']
                is_expired = item['is_expired']
                
                if is_expired:
                    # Display for expired contract
                    expiration_date_str = data.get('expiration_date', data['current_date']).strftime('%d/%m')
                    print(f"{comp_name:<12} {market:<8} {data['current_price']:>8.3f}      {'-':<8} {'-':<8} {'EXPIRED':<10} {expiration_date_str:<12} "
                          f"{data['contracts_coverage']:<6}")
                else:
                    # Display for active contract with variation
                    color_circle = self._get_variation_color(data['variation_pct'])
                    current_date_str = data['current_date'].strftime('%d/%m')
                    previous_date_str = data['previous_date'].strftime('%d/%m')
                    dates_display = f"{previous_date_str}→{current_date_str}"
                    
                    print(f"{comp_name:<12} {market:<8} {data['current_price']:>8.3f}      "
                          f"{data['variation_abs']:>+7.3f} "
                          f"{color_circle} {data['variation_pct']:>+6.2f}% "
                          f"{'ACTIVE':<10} {dates_display:<12} "
                          f"{data['contracts_coverage']:<6}")
            
            print(f"\n🎨 Color legend:")
            print("🟢 Strong increase (≥5%)  🟡 Moderate increase (2-5%)  🟠 Low increase (<2%)")
            print("🔴 Strong decline (≥5%)  🟤 Moderate decline (2-5%)  🟣 Low decline (<2%)")
            print("📌 EXPIRED : Price fixed on the expiration date")
            print("📌 ACTIVE  : Calculated variations over the period")


    def standard_variation(self, markets: Union[str, List[str], None] = None, 
                           periods: Union[str, List[str], None] = None,
                           contracts: Union[str, List[str], None] = None,
                           display: bool = True):
        """
        Calculates daily, weekly, and monthly variations for standard contracts
        with expiration management - displays only the final price for expired contracts
        """
        self._load_data()
        
        # Markets parameter normalization
        if markets is None:
            markets = ['TTF', 'PEG']
        elif isinstance(markets, str):
            markets = [markets.upper()]
        else:
            markets = [m.upper() for m in markets]
        
        # Periods parameter normalization
        available_periods = ['daily', 'weekly', 'monthly']
        
        if periods is None:
            periods = available_periods
        elif isinstance(periods, str):
            if periods.lower() in available_periods:
                periods = [periods.lower()]
            else:
                print(f"⚠️ Period '{periods}' not recognized. Using default periods.")
                periods = available_periods
        else:
            # Filter valid periods
            valid_periods = [p.lower() for p in periods if p.lower() in available_periods]
            if valid_periods:
                periods = valid_periods
            else:
                print("⚠️ No valid period specified. Using default periods.")
                periods = available_periods
        
        # Default list of standard contracts
        default_standard_contracts = [
            'Oct25', 'Nov25', 'Dec25', 'Jan26', 'Feb26', 'Mar26',
            'Apr26', 'May26', 'Jun26', 'Jul26', 'Aug26', 'Sep26',
            'Oct26', 'Nov26', 'Dec26', 'Jan27', 'Feb27', 'Mar27'
        ]
        
        # Contracts parameter normalization
        if contracts is None:
            contracts_to_analyze = default_standard_contracts
        else:
            if isinstance(contracts, str):
                contracts_to_analyze = [contracts]
            else:
                contracts_to_analyze = list(contracts)
        
        if not contracts_to_analyze:
            print("❌ No valid standard contract to analyze.")
            return {}
        
        variations_results = {}
        
        for period_name in periods:
            variations_results[period_name] = {}
            
            for market in markets:
                variations_results[period_name][market] = {}
                
                for contract in contracts_to_analyze:
                    contract_id = f"{market}{contract}"
                    
                    # Check if contract exists in data
                    if contract_id not in self._data:
                        continue
                    
                    # Calculate current price with closing time management
                    current_price_data = self._get_contract_price(market, contract, 'standard')
                    if not current_price_data:
                        continue
                    
                    # Check if contract is expired
                    df = self._data[contract_id]
                    is_expired = self._is_contract_expired(df, current_price_data['date'])
                    
                    if is_expired:
                        # Expired contract: display only final price without variation
                        variations_results[period_name][market][contract] = {
                            'current_price': current_price_data['price'],
                            'previous_price': current_price_data['price'],
                            'variation_abs': 0.0,
                            'variation_pct': 0.0,
                            'current_date': current_price_data['date'],
                            'previous_date': current_price_data['date'],
                            'is_expired': True,
                            'expiration_date': current_price_data['date']
                        }
                    else:
                        # Active contract: calculate normal variation
                        previous_price_data = self._get_contract_price_at_period(
                            market, contract, 'standard', period_name
                        )
                        
                        if not previous_price_data:
                            continue
                        
                        # Variation calculation
                        current_price = current_price_data['price']
                        previous_price = previous_price_data['price']
                        
                        variation_abs = current_price - previous_price
                        variation_pct = (variation_abs / previous_price) * 100 if previous_price != 0 else 0
                        
                        variations_results[period_name][market][contract] = {
                            'current_price': current_price,
                            'previous_price': previous_price,
                            'variation_abs': variation_abs,
                            'variation_pct': variation_pct,
                            'current_date': current_price_data['date'],
                            'previous_date': previous_price_data['date'],
                            'is_expired': False
                        }
        
        if display:
            self._display_standard_variations(variations_results)
        
        
    
    
    def _is_contract_expired(self, df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
        """
        Check if a contract is expired (no new data for more than 5 business days)
        """
        if df.empty:
            return True
        
        # Get last available date in data
        last_date = df['Date'].max()
        
        # If difference is more than 7 calendar days, consider expired
        days_diff = (current_date - last_date).days
        
        return days_diff > 7
    
    
    def _display_standard_variations(self, variations_results: Dict):
        """Display standard contract variations in formatted way"""
        
        period_names = {
            'daily': 'DAILY VARIATIONS (STANDARD CONTRACTS)',
            'weekly': 'WEEKLY VARIATIONS (STANDARD CONTRACTS)', 
            'monthly': 'MONTHLY VARIATIONS (STANDARD CONTRACTS)'
        }
        
        for period, period_display in period_names.items():
            if period not in variations_results:
                continue
                
            print("\n" + "=" * 100)
            print(f"{period_display}".center(100))
            print("=" * 100)
            
            # Table header
            print(f"{'Contract':<12} {'Market':<8} {'Price (€/MWh)':<12} {'Var. €':<8} {'Var. %':<8} {'Status':<10} {'Date':<12}")
            print("-" * 100)
            
            # Collect all variations for this period for sorting
            all_contracts = []
            for market, contracts_data in variations_results[period].items():
                if not contracts_data:
                    continue
                for contract_name, data in contracts_data.items():
                    all_contracts.append({
                        'market': market,
                        'contract': contract_name,
                        'data': data,
                        'is_expired': data.get('is_expired', False)
                    })
            
            # Sort: expired contracts first, then by name
            all_contracts.sort(key=lambda x: (not x['is_expired'], x['contract']))
            
            for item in all_contracts:
                data = item['data']
                market = item['market']
                contract_name = item['contract']
                is_expired = item['is_expired']
                
                if is_expired:
                    # Display for expired contract
                    expiration_date_str = data.get('expiration_date', data['current_date']).strftime('%d/%m')
                    print(f"{contract_name:<12} {market:<8} {data['current_price']:>8.3f}      {'-':<8} {'-':<8} {'EXPIRED':<10} {expiration_date_str:<12}")
                else:
                    # Display for active contract with variation
                    color_circle = self._get_variation_color(data['variation_pct'])
                    current_date_str = data['current_date'].strftime('%d/%m')
                    previous_date_str = data['previous_date'].strftime('%d/%m')
                    dates_display = f"{previous_date_str}→{current_date_str}"
                    
                    print(f"{contract_name:<12} {market:<8} {data['current_price']:>8.3f}      "
                          f"{data['variation_abs']:>+7.3f} "
                          f"{color_circle} {data['variation_pct']:>+6.2f}% "
                          f"{'ACTIVE':<10} {dates_display:<12}")
            
            print(f"\n🎨 Color legend:")
            print("🟢 Strong increase (≥5%)  🟡 Moderate increase (2-5%)  🟠 Low increase (<2%)")
            print("🔴 Strong decline (≥5%)  🟤 Moderate decline (2-5%)  🟣 Low decline (<2%)")
            print("📌 EXPIRED : Price fixed on the expiration date")
            print("📌 ACTIVE  : Calculated variations over the period")
            
#================================= PRICE CURVE ==============================================================#
    def plot_curves(self, curve_type: str = 'standard', past_data: bool = False, 
                    figsize: Tuple[int, int] = (14, 8)):
        """
        Displays TTF and PEG curves on the same graph
        
        Args:
            curve_type: Curve type ('standard' or 'composite')
                       - 'standard': Monthly contracts from Oct25 to Mar27
                       - 'composite': Quarterly contracts only
            past_data: If True, also display the curves from the day before yesterday as dotted lines.
            figsize: Figure size (width, height)
        """
        self._load_data()
        
        # Define contracts according to requested type
        if curve_type.lower() == 'standard':
            # Standard monthly contracts from Oct25 to Mar27
            base_contracts = ['Oct25', 'Nov25', 'Dec25', 'Jan26', 'Feb26', 'Mar26', 
                             'Apr26', 'May26', 'Jun26', 'Jul26', 'Aug26', 'Sep26',
                             'Oct26', 'Nov26', 'Dec26', 'Jan27', 'Feb27', 'Mar27']
            title_suffix = "Standard Monthly Contracts (Oct25-Mar27)"
        elif curve_type.lower() == 'composite':
            # Quarterly contracts only
            base_contracts = ['Q425', 'Q126', 'Q226', 'Q326', 'Q426']
            title_suffix = "Quarterly Contracts"
        else:
            print("❌ Curve type not recognized. Use 'standard' or 'composite'")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Define colors
        colors = {
            'TTF': '#2E86AB',      # Dark blue
            'PEG': '#A23B72',      # Violet/Pink
            'TTF_past': '#7FB3D3', # Light blue for past data
            'PEG_past': '#D4A5C4'  # Light pink for past data
        }
        
        # Find reference dates
        today = pd.Timestamp.today().normalize()
        current_target_date = None
        past_target_date = None
        
        # Find last available date (previous day)
        for market in ['TTF', 'PEG']:
            for contract in base_contracts:
                contract_id = f"{market}{contract}"
                if contract_id in self._data:
                    df = self._data[contract_id]
                    df_filtered = df[df['Date'] < today].sort_values('Date', ascending=False)
                    if not df_filtered.empty:
                        candidate_date = df_filtered['Date'].iloc[0]
                        if current_target_date is None or candidate_date > current_target_date:
                            current_target_date = candidate_date
        
        if current_target_date is None:
            print("❌ No data found for specified contracts")
            return
        
        # For past data, look for day before yesterday
        if past_data:
            for market in ['TTF', 'PEG']:
                for contract in base_contracts:
                    contract_id = f"{market}{contract}"
                    if contract_id in self._data:
                        df = self._data[contract_id]
                        df_filtered = df[df['Date'] < current_target_date].sort_values('Date', ascending=False)
                        if not df_filtered.empty:
                            candidate_date = df_filtered['Date'].iloc[0]
                            if past_target_date is None or candidate_date > past_target_date:
                                past_target_date = candidate_date
        
        # Function to get prices at given date
        def get_prices_at_date(target_date, contracts_list):
            ttf_prices = []
            peg_prices = []
            x_labels = []
            
            for contract in contracts_list:
                ttf_price = None
                peg_price = None
                
                # TTF price
                ttf_contract = f"TTF{contract}"
                if curve_type.lower() == 'composite' and contract in self.composite_contracts:
                    # For composite contracts, calculate average
                    monthly_contracts = [f"TTF{m}" for m in self.composite_contracts[contract]]
                    composite_data = self._calculate_composite_price_at_date_specific(monthly_contracts, target_date)
                    if composite_data:
                        ttf_price = composite_data['price']
                elif ttf_contract in self._data:
                    df = self._data[ttf_contract]
                    df_filtered = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
                    if not df_filtered.empty:
                        ttf_price = df_filtered['Close'].iloc[0]
                
                # PEG price
                peg_contract = f"PEG{contract}"
                if curve_type.lower() == 'composite' and contract in self.composite_contracts:
                    # For composite contracts, calculate average
                    monthly_contracts = [f"PEG{m}" for m in self.composite_contracts[contract]]
                    composite_data = self._calculate_composite_price_at_date_specific(monthly_contracts, target_date)
                    if composite_data:
                        peg_price = composite_data['price']
                elif peg_contract in self._data:
                    df = self._data[peg_contract]
                    df_filtered = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
                    if not df_filtered.empty:
                        peg_price = df_filtered['Close'].iloc[0]
                
                # Add prices if at least one is available
                if ttf_price is not None or peg_price is not None:
                    ttf_prices.append(ttf_price)
                    peg_prices.append(peg_price)
                    x_labels.append(contract)
            
            return ttf_prices, peg_prices, x_labels
        
        # Get current prices (previous day)
        ttf_current, peg_current, labels = get_prices_at_date(current_target_date, base_contracts)
        
        if not labels:
            print("❌ No data available for specified contracts")
            return
        
        x_pos = range(len(labels))
        
        # Plot current curves
        ttf_line = plt.plot(x_pos, ttf_current, 'o-', color=colors['TTF'], 
                           linewidth=2.5, markersize=6, label=f'TTF ({current_target_date.strftime("%d/%m/%Y")})')
        peg_line = plt.plot(x_pos, peg_current, 's-', color=colors['PEG'], 
                           linewidth=2.5, markersize=6, label=f'PEG ({current_target_date.strftime("%d/%m/%Y")})')
        
        # Plot past curves if requested
        if past_data and past_target_date:
            ttf_past, peg_past, _ = get_prices_at_date(past_target_date, base_contracts)
            
            if len(ttf_past) == len(ttf_current):
                plt.plot(x_pos, ttf_past, 'o--', color=colors['TTF_past'], 
                        linewidth=2, markersize=4, alpha=0.7, 
                        label=f'TTF ({past_target_date.strftime("%d/%m/%Y")})')
            
            if len(peg_past) == len(peg_current):
                plt.plot(x_pos, peg_past, 's--', color=colors['PEG_past'], 
                        linewidth=2, markersize=4, alpha=0.7,
                        label=f'PEG ({past_target_date.strftime("%d/%m/%Y")})')
        
        # Graph configuration
        plt.title(f'TTF vs PEG Curves - {title_suffix}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Contracts', fontsize=12, fontweight='bold')
        plt.ylabel('Price (€/MWh)', fontsize=12, fontweight='bold')
        
        # X-axis configuration
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        
        # Grid and legend
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Improve layout
        plt.tight_layout()
        
        # Display statistics
        print(f"\n📊 CURVE STATISTICS ({current_target_date.strftime('%d/%m/%Y')})")
        print("=" * 70)
        
        # Calculate statistics for TTF and PEG
        ttf_valid = [p for p in ttf_current if p is not None]
        peg_valid = [p for p in peg_current if p is not None]
        
        if ttf_valid:
            print(f"TTF - Min: {min(ttf_valid):.3f} €/MWh | Max: {max(ttf_valid):.3f} €/MWh | Average: {np.mean(ttf_valid):.3f} €/MWh")
        
        if peg_valid:
            print(f"PEG - Min: {min(peg_valid):.3f} €/MWh | Max: {max(peg_valid):.3f} €/MWh | Average: {np.mean(peg_valid):.3f} €/MWh")
        
        # Calculate spreads if both curves available
        if ttf_valid and peg_valid and len(ttf_valid) == len(peg_valid):
            spreads = [(t - p) if (t is not None and p is not None) else None 
                      for t, p in zip(ttf_current, peg_current)]
            spreads_valid = [s for s in spreads if s is not None]
            
            if spreads_valid:
                print(f"Spread (TTF-PEG) - Min: {min(spreads_valid):.3f} €/MWh | Max: {max(spreads_valid):.3f} €/MWh | Average: {np.mean(spreads_valid):.3f} €/MWh")
        
        plt.show()

    def _calculate_composite_price_at_date_specific(self, contracts: List[str], target_date: pd.Timestamp) -> Optional[Dict]:
        """
        Calculates the composite price on a specific date
        """
        available_contracts = [c for c in contracts if c in self._data]
        if not available_contracts:
            return None
        
        prices = []
        actual_date = None
        individual_prices = {}
        used_contracts = []
        
        for contract in available_contracts:
            df = self._data[contract]
            
            # Find price closest to target_date (backwards)
            df_filtered = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
            
            if not df_filtered.empty:
                price = df_filtered['Close'].iloc[0]
                date = df_filtered['Date'].iloc[0]
                
                prices.append(price)
                individual_prices[contract] = price
                used_contracts.append(contract)
                
                if actual_date is None or date > actual_date:
                    actual_date = date
        
        if not prices:
            return None
        
        return {
            'price': np.mean(prices),
            'date': actual_date,
            'contracts': used_contracts,
            'individual_prices': individual_prices
        }

#================================= LOCATION SPREAD ==============================================================#
    def location_spread(self, type: str, markets: List[str] = None, contracts: Union[str, List[str]] = None,
                       periods: Union[str, List[str]] = None, graph: bool = True, 
                       volatility_window: Optional[int] = None):        
        """
        Calculates location spreads between different markets (e.g., TTF vs. PEG)
        
        Args:
            type: Contract type ('standard' or 'composite') - REQUIRED
            markets: List of markets to compare (default: ['TTF', 'PEG'])
            contracts: Contract(s) to analyze (e.g., 'Oct25' or ['Oct25', 'Nov25'])
            periods: Period(s) for variations ('daily', 'weekly', 'monthly')
            graph: Display graph with moving averages (default: True)
            volatility_window: Window in days for volatility/return analysis (e.g., 7, 30)
        
        Returns:
            dict: Spread analysis results
        """
        if type.lower() not in ['standard', 'composite']:
            print("❌ Parameter 'type' must be 'standard' or 'composite'")
            return {}
        
        self._load_data()
        
        # Default parameters
        if markets is None:
            markets = ['TTF', 'PEG']
        
        if len(markets) != 2:
            print("❌ Exactly 2 markets must be specified to calculate a spread")
            return {}
        
        market1, market2 = markets[0].upper(), markets[1].upper()
        
        # Define contracts according to type
        if type.lower() == 'standard':
            default_contracts = ['Oct25', 'Nov25', 'Dec25', 'Jan26', 'Feb26', 'Mar26', 
                               'Apr26', 'May26', 'Jun26', 'Jul26', 'Aug26', 'Sep26',
                               'Oct26', 'Nov26', 'Dec26', 'Jan27', 'Feb27', 'Mar27']
        else:  # composite
            default_contracts = ['Q425', 'Q126', 'Q226', 'Q326', 'Q426']
        
        if contracts is None:
            contracts_to_analyze = default_contracts
        else:
            if isinstance(contracts, str):
                contracts_to_analyze = [contracts]
            else:
                contracts_to_analyze = contracts
        
        # Period parameters
        if periods is None:
            periods = ['daily', 'weekly', 'monthly']
        elif isinstance(periods, str):
            periods = [periods.lower()]
        else:
            periods = [p.lower() for p in periods]
        
        # Results
        results = {}
        
        print(f"\n🔄 LOCATION SPREAD ANALYSIS ({market1} vs {market2})")
        print("=" * 80)
        
        for contract in contracts_to_analyze:
            print(f"\n📊 Contract: {contract}")
            print("-" * 50)
            
            # Calculate current spread and variations
            spread_data = self._calculate_spread_analysis(
                market1, market2, contract, type.lower(), periods
            )
            
            if not spread_data:
                print(f"❌ Insufficient data for {contract}")
                continue
            
            results[contract] = spread_data
            
            # Display results
            self._display_spread_results(contract, spread_data, market1, market2)
            
            # Graph with moving averages if requested
            if graph:
                self._plot_spread_with_ma(
                    market1, market2, contract, type.lower(), 
                    ma_periods=[50, 100]
                )
            
            # Volatility and returns analysis if requested
            if volatility_window is not None:
                vol_analysis = self._analyze_spread_volatility_and_returns(
                    market1, market2, contract, type.lower(), volatility_window
                )
                if vol_analysis:
                    results[contract]['volatility_analysis'] = vol_analysis
                    self._plot_volatility_and_returns(vol_analysis, contract, market1, market2, volatility_window)
        
        

    def _calculate_spread_analysis(self, market1: str, market2: str, contract: str, 
                                  contract_type: str, periods: List[str]) -> Optional[Dict]:
        """
        Calculates the current spread and its variations over different periods
        """
        # Get current prices
        current_price1 = self._get_contract_price(market1, contract, contract_type)
        current_price2 = self._get_contract_price(market2, contract, contract_type)
        
        if not current_price1 or not current_price2:
            return None
        
        current_spread = current_price1['price'] - current_price2['price']
        current_date = current_price1['date']
        
        results = {
            'current_spread': current_spread,
            'current_date': current_date,
            'market1_price': current_price1['price'],
            'market2_price': current_price2['price'],
            'variations': {}
        }
        
        # Calculate variations for each period
        for period in periods:
            past_price1 = self._get_contract_price_at_period(
                market1, contract, contract_type, period
            )
            past_price2 = self._get_contract_price_at_period(
                market2, contract, contract_type, period
            )
            
            if past_price1 and past_price2:
                past_spread = past_price1['price'] - past_price2['price']
                variation_abs = current_spread - past_spread
                variation_pct = (variation_abs / abs(past_spread)) * 100 if past_spread != 0 else 0
                
                results['variations'][period] = {
                    'past_spread': past_spread,
                    'past_date': past_price1['date'],
                    'variation_abs': variation_abs,
                    'variation_pct': variation_pct
                }
        
        return results

    def _calculate_composite_price(self, contracts: List[str]) -> Optional[Dict]:
        """
        Calculates the composite closing price on the last full trading day (D-1 or earlier)
        """
        available_contracts = [c for c in contracts if c in self._data]
        if not available_contracts:
            return None
        
        # Get today's date (normalized)
        today = pd.Timestamp.today().normalize()
        
        prices = []
        actual_date = None
        individual_prices = {}
        
        for contract in available_contracts:
            df = self._data[contract]
            if df.empty:
                continue
            
            # Filter dates strictly before today (to get previous day closing)
            df_past = df[df['Date'] < today].sort_values('Date', ascending=False)
            
            if df_past.empty:
                # If no data before today, take last available
                df_past = df.sort_values('Date', ascending=False)
                if df_past.empty:
                    continue
            
            latest_row = df_past.iloc[0]
            
            price = latest_row['Close']
            date = latest_row['Date']
            
            prices.append(price)
            individual_prices[contract] = price
            
            # Keep most recent date among all contracts
            if actual_date is None or date > actual_date:
                actual_date = date
        
        if not prices:
            return None
        
        return {
            'price': np.mean(prices),
            'date': actual_date,
            'contracts': available_contracts,
            'individual_prices': individual_prices
        }
    
    def _get_contract_price(self, market: str, contract: str, contract_type: str) -> Optional[Dict]:
        """
        Gets the current price of a contract (standard or composite) from the last full trading day
        """
        try:
            # Get today's date (normalized)
            today = pd.Timestamp.today().normalize()
            
            if contract_type == 'composite':
                # Check if composite contract exists
                if contract not in self.composite_contracts:
                    return None
                
                monthly_contracts = self.composite_contracts[contract]
                prices = []
                latest_date = None
                
                for monthly_contract in monthly_contracts:
                    contract_id = f"{market}{monthly_contract}"
                    
                    if contract_id in self._data:
                        df = self._data[contract_id]
                        if len(df) > 0:
                            # Filter dates strictly before today (to get previous day closing)
                            df_past = df[df['Date'] < today].sort_values('Date', ascending=False)
                            
                            if df_past.empty:
                                # If no data before today, take last available
                                df_past = df.sort_values('Date', ascending=False)
                                if df_past.empty:
                                    continue
                            
                            latest_row = df_past.iloc[0]
                            price = latest_row['Close']
                            date = latest_row['Date']
                            
                            prices.append(price)
                            
                            # Keep most recent date among all contracts
                            if latest_date is None or date > latest_date:
                                latest_date = date
                
                if prices and latest_date:
                    avg_price = sum(prices) / len(prices)
                    print(f"✅ Composite price {contract}: {avg_price:.3f} €/MWh ({len(prices)}/{len(monthly_contracts)} contracts)")
                    return {
                        'price': avg_price,
                        'date': latest_date,
                        'components': len(prices),
                        'component_prices': prices
                    }
                else:
                    print(f"❌ No price found for composite {contract}")
                    return None
            
            else:  # standard
                contract_id = f"{market}{contract}"
                if contract_id in self._data:
                    df = self._data[contract_id]
                    if len(df) > 0:
                        # Filter dates strictly before today (to get previous day closing)
                        df_past = df[df['Date'] < today].sort_values('Date', ascending=False)
                        
                        if df_past.empty:
                            # If no data before today, take last available
                            df_past = df.sort_values('Date', ascending=False)
                            if df_past.empty:
                                return None
                        
                        latest_row = df_past.iloc[0]
                        return {
                            'price': latest_row['Close'],
                            'date': latest_row['Date']
                        }
                print(f"❌ Standard contract not found: {contract_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error retrieving price for {market}{contract}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_contract_price_at_period(self, market: str, contract: str, contract_type: str, 
                                    period: str) -> Optional[Dict]:
        """
        Obtains the price of a contract from a previous period
        """
        try:
            if contract_type == 'composite':
                if contract not in self.composite_contracts:
                    return None
                
                monthly_contracts = self.composite_contracts[contract]
                prices = []
                target_date = None
                
                # Determine offset according to period
                if period == 'daily':
                    days_offset = 2
                elif period == 'weekly':
                    days_offset = 8
                elif period == 'monthly':
                    days_offset = 29
                else:
                    days_offset = 1
                
                for monthly_contract in monthly_contracts:
                    contract_id = f"{market}{monthly_contract}"
                    if contract_id in self._data:
                        df = self._data[contract_id].copy()
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date')
                        
                        if len(df) > 0:
                            current_date = df.iloc[-1]['Date']
                            past_date = current_date - pd.Timedelta(days=days_offset)
                            
                            # Find closest date before past_date
                            past_data = df[df['Date'] <= past_date]
                            if len(past_data) > 0:
                                closest_row = past_data.iloc[-1]
                                prices.append(closest_row['Close'])
                                
                                if target_date is None or closest_row['Date'] > target_date:
                                    target_date = closest_row['Date']
                
                if prices and target_date:
                    avg_price = sum(prices) / len(prices)
                    return {
                        'price': avg_price,
                        'date': target_date,
                        'components': len(prices)
                    }
                return None
            
            else:  # standard contract
                contract_id = f"{market}{contract}"
                if contract_id in self._data:
                    df = self._data[contract_id]
                    
                    # ADD THIS LINE:
                    today = pd.Timestamp.today().normalize()
                    
                    # First, get current reference date (D-1)
                    df_current = df[df['Date'] < today].sort_values('Date', ascending=False)
                    if df_current.empty:
                        return None
                    
                    current_ref_date = df_current.iloc[0]['Date']
                    
                    # For daily: find the previous business day (D-2)
                    if period == 'daily':
                        # Find the date before current_ref_date in the data
                        df_before = df[df['Date'] < current_ref_date].sort_values('Date', ascending=False)
                        if not df_before.empty:
                            previous_row = df_before.iloc[0]
                            return {
                                'price': previous_row['Close'],
                                'date': previous_row['Date']
                            }
                        else:
                            return None
                    
                    # For weekly and monthly, use the existing logic
                    elif period == 'weekly':
                        target_date = current_ref_date - pd.Timedelta(days=7)
                    elif period == 'monthly':
                        target_date = current_ref_date - pd.Timedelta(days=29)
                    else:
                        target_date = current_ref_date - pd.Timedelta(days=1)
                    
                    # Find closest date to target_date (before or equal)
                    df_past = df[df['Date'] <= target_date].sort_values('Date', ascending=False)
                    if not df_past.empty:
                        closest_row = df_past.iloc[0]
                        return {
                            'price': closest_row['Close'],
                            'date': closest_row['Date']
                        }
                return None
                
        except Exception as e:
            print(f"❌ Error in _get_contract_price_at_period: {e}")
            return None
            
    def _display_spread_results(self, contract: str, spread_data: Dict, market1: str, market2: str):
        """
        Displays the results of the spread analysis
        """
        current_spread = spread_data['current_spread']
        current_date = spread_data['current_date']
        
        print(f"💰 Current spread ({market1}-{market2}): {current_spread:+.3f} €/MWh")
        print(f"   └─ {market1}: {spread_data['market1_price']:.3f} €/MWh")
        print(f"   └─ {market2}: {spread_data['market2_price']:.3f} €/MWh")
        
        if spread_data['variations']:
            print("\n📊 Spread variations:")
            for period, var_data in spread_data['variations'].items():
                period_names = {'daily': 'Day', 'weekly': 'Week', 'monthly': 'Month'}
                period_name = period_names.get(period, period)
                
                color_circle = self._get_variation_color(var_data['variation_pct'])
                
                print(f"   • {period_name:<8}: {var_data['variation_abs']:+.3f} €/MWh "
                      f"{color_circle} {var_data['variation_pct']:+.1f}% "
                      f"(since {var_data['past_date'].strftime('%d/%m')})")
    
    def _plot_spread_with_ma(self, market1: str, market2: str, contract: str, 
                           contract_type: str, ma_periods: List[int] = [50, 100]):
        """
        Plot the spread graph with moving averages
        
        """
        # Get complete historical spread data
        spread_series = self._get_historical_spread(market1, market2, contract, contract_type)
        
        if spread_series is None or len(spread_series) < max(ma_periods):
            print(f"⚠️  Insufficient historical data for MA{ma_periods} on {contract}")
            print(f"   (Available: {len(spread_series) if spread_series is not None else 0} days, required: {max(ma_periods)})")
            return
        
        # Calculate moving averages
        ma_data = {}
        for period in ma_periods:
            if len(spread_series) >= period:
                ma_data[f'MA{period}'] = spread_series.rolling(window=period).mean()
        
        # Create graph
        plt.figure(figsize=(14, 8))
        
        # Plot spread
        plt.plot(spread_series.index, spread_series.values, 
                color='#2E86AB', linewidth=1.5, label=f'Spread {market1}-{market2}', alpha=0.8)
        
        # Plot moving averages
        colors_ma = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (ma_name, ma_series) in enumerate(ma_data.items()):
            plt.plot(ma_series.index, ma_series.values, 
                    color=colors_ma[i % len(colors_ma)], linewidth=2, 
                    label=ma_name, alpha=0.9)
        
        # Graph configuration
        title_suffix = f"Composite {contract}" if contract_type == 'composite' else f"Standard {contract}"
        plt.title(f'Location Spread {market1} vs {market2} - {title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Spread (€/MWh)', fontsize=12, fontweight='bold')
        
        # Zero line
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Date rotation
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Graph statistics
        current_spread = spread_series.iloc[-1]
        min_spread = spread_series.min()
        max_spread = spread_series.max()
        avg_spread = spread_series.mean()
        
    
        
        plt.show()
    
    def _get_historical_spread(self, market1: str, market2: str, contract: str, 
                              contract_type: str) -> Optional[pd.Series]:
        """
        Obtains the complete historical series of the spread
        """
        if contract_type == 'composite' and contract in self.composite_contracts:
            # For composites, calculate composite spread over entire history
            monthly_contracts1 = [f"{market1}{m}" for m in self.composite_contracts[contract]]
            monthly_contracts2 = [f"{market2}{m}" for m in self.composite_contracts[contract]]
            
            # Verify all contracts exist
            if not all(c in self._data for c in monthly_contracts1 + monthly_contracts2):
                return None
            
            # Find dates common to all contracts
            all_dates = None
            for contract_id in monthly_contracts1 + monthly_contracts2:
                df = self._data[contract_id].set_index('Date')
                if all_dates is None:
                    all_dates = set(df.index)
                else:
                    all_dates = all_dates.intersection(set(df.index))
            
            if not all_dates:
                return None
            
            all_dates = sorted(list(all_dates))
            
            # Calculate composite spread for each date
            spread_values = []
            for date in all_dates:
                price1 = sum(self._data[c].set_index('Date').loc[date, 'Close'] 
                           for c in monthly_contracts1) / len(monthly_contracts1)
                price2 = sum(self._data[c].set_index('Date').loc[date, 'Close'] 
                           for c in monthly_contracts2) / len(monthly_contracts2)
                spread_values.append(price1 - price2)
            
            return pd.Series(spread_values, index=all_dates)
        
        else:
            # For standard contracts
            contract_id1 = f"{market1}{contract}"
            contract_id2 = f"{market2}{contract}"
            
            if contract_id1 not in self._data or contract_id2 not in self._data:
                return None
            
            df1 = self._data[contract_id1].set_index('Date')
            df2 = self._data[contract_id2].set_index('Date')
            
            # Align dates
            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) == 0:
                return None
            
            df1_aligned = df1.loc[common_dates]
            df2_aligned = df2.loc[common_dates]
            
            return df1_aligned['Close'] - df2_aligned['Close']
    
    def _analyze_spread_volatility_and_returns(self, market1: str, market2: str, contract: str, 
                                             contract_type: str, window: int) -> Optional[Dict]:         
        """
        Analyzes volatility and spread returns on a rolling window
        ONLY USES DATA UP TO D-1 (previous business day)
        """
        # Get complete historical spread series
        spread_series = self._get_historical_spread(market1, market2, contract, contract_type)
        
        if spread_series is None or len(spread_series) < window + 1:
            return None
        
        # Sort by date and filter out today's data
        spread_series = spread_series.sort_index()
        today = pd.Timestamp.today().normalize()
        spread_series = spread_series[spread_series.index < today]  # ONLY DATA BEFORE TODAY
        
        if len(spread_series) < window + 1:
            return None
        
        # Calculate daily spread returns
        returns = spread_series.pct_change().dropna()
        
        # Calculate rolling volatility (NOT annualized)
        rolling_volatility = returns.rolling(window=window).std()  # Not annualized
        
        # Take last observations according to window for some analyses
        recent_data = spread_series.tail(window)
        recent_returns = returns.tail(window)
        recent_volatility = rolling_volatility.tail(window)
        
        return {
            'spread_values': spread_series,  # All data now
            'returns': returns,  # All returns
            'rolling_volatility': rolling_volatility,  # All volatility
            'current_volatility': recent_volatility.iloc[-1] if not recent_volatility.empty else None,
            'avg_return': returns.mean(),  # Average over entire period
            'return_volatility': returns.std(),  # Volatility over entire period
            'min_spread': spread_series.min(),
            'max_spread': spread_series.max(),
            'window': window,
            'recent_window_data': {  # Keep recent data separately
                'spread_values': recent_data,
                'returns': recent_returns,
                'rolling_volatility': recent_volatility
            }
        }
    
    def _plot_volatility_and_returns(self, vol_analysis: Dict, contract: str, 
                                    market1: str, market2: str, window: int):        
        """
        Plots volatility and return graphs
        """
        # Helper function to detect and filter outliers
        def filter_outliers(data, method='iqr', threshold=3):
            """
            Filters outliers using the IQR or Z-score method
            Returns a Boolean mask for valid values
            """
            data_clean = data[np.isfinite(data)]
            if len(data_clean) == 0:
                return np.zeros(len(data), dtype=bool)
            
            if method == 'iqr':
                Q1 = np.percentile(data_clean, 25)
                Q3 = np.percentile(data_clean, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            else:  # z-score
                mean = np.mean(data_clean)
                std = np.std(data_clean)
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            
            return (data >= lower_bound) & (data <= upper_bound) & np.isfinite(data)
        
        # Create 2x2 layout
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
        
        # 1. Complete spread evolution (top left)
        ax1 = plt.subplot(gs[0, 0])
        spread_data = vol_analysis['spread_values']
        spread_clean_mask = np.isfinite(spread_data.values)
        
        if np.sum(spread_clean_mask) > 0:
            ax1.plot(spread_data.index[spread_clean_mask], spread_data.values[spread_clean_mask],
                    color='#2E86AB', linewidth=2, alpha=0.8)
            ax1.set_title(f'Spread Evolution {market1}-{market2}\n(Complete)', 
                         fontweight='bold', fontsize=12)
            ax1.set_ylabel('Spread (€/MWh)', fontweight='bold', fontsize=10)
            ax1.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            
            current_spread = spread_data.values[spread_clean_mask][-1]
            ax1.axhline(y=current_spread, color='red', linestyle=':', alpha=0.8, 
                       label=f'Current: {current_spread:.3f} €/MWh')
            ax1.legend(loc='best', fontsize=8)
        
        # 2. Recent spread evolution (top right)
        ax2 = plt.subplot(gs[0, 1])
        recent_data = vol_analysis['recent_window_data']['spread_values']
        recent_clean_mask = np.isfinite(recent_data.values)
        
        if np.sum(recent_clean_mask) > 0:
            ax2.plot(recent_data.index[recent_clean_mask], recent_data.values[recent_clean_mask],
                    color='#E74C3C', linewidth=2.5, marker='o', markersize=3)
            ax2.set_title(f'Recent Spread Evolution\n({window} last days)', 
                         fontweight='bold', fontsize=12)
            ax2.set_ylabel('Spread (€/MWh)', fontweight='bold', fontsize=10)
            ax2.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 3. Returns distribution WITHOUT OUTLIERS (middle left)
        ax3 = plt.subplot(gs[1, 0])
        returns_pct = vol_analysis['returns'].values * 100
        
        # Filter NaN/Inf
        returns_finite_mask = np.isfinite(returns_pct)
        returns_pct_finite = returns_pct[returns_finite_mask]
        
        # Detect and filter outliers
        if len(returns_pct_finite) > 0:
            outliers_mask = filter_outliers(returns_pct_finite, method='iqr', threshold=3)
            returns_no_outliers = returns_pct_finite[outliers_mask]
            n_outliers = len(returns_pct_finite) - len(returns_no_outliers)
            
            if len(returns_no_outliers) > 0:
                n, bins, patches = ax3.hist(returns_no_outliers, 
                                           bins=min(30, max(10, len(returns_no_outliers)//10)), 
                                           alpha=0.7, color='#4ECDC4', edgecolor='black', 
                                           density=True)
                
                # Color negative bars in red
                for patch, left_edge in zip(patches, bins):
                    if left_edge < 0:
                        patch.set_facecolor('#FF6B6B')
                        patch.set_alpha(0.7)
                
                ax3.set_title(f'Daily Returns Distribution\n(Without outliers - {n_outliers} excluded)', 
                             fontweight='bold', fontsize=12)
                ax3.set_xlabel('Return (%)', fontweight='bold', fontsize=10)
                ax3.set_ylabel('Density', fontweight='bold', fontsize=10)
                ax3.grid(True, which='major', axis='y', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
                
                # Statistics
                mean_return_no_outliers = np.mean(returns_no_outliers)
                ax3.axvline(x=mean_return_no_outliers, color='red', linestyle='--', linewidth=2,
                           label=f'Average: {mean_return_no_outliers:.3f}%')
                
                # Skewness and kurtosis
                from scipy import stats
                if len(returns_no_outliers) > 2:
                    skewness = stats.skew(returns_no_outliers)
                    kurtosis = stats.kurtosis(returns_no_outliers)
                    ax3.text(0.02, 0.98, 
                            f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}\nOutliers: {n_outliers}', 
                            transform=ax3.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)
                
                ax3.legend(fontsize=8)
                
                # Info on outliers
                if n_outliers > 0:
                    outliers_values = returns_pct_finite[~outliers_mask]
                    print(f"\n⚠️  {n_outliers} outliers detected in returns:")
                    print(f"   Min outlier: {outliers_values.min():.2f}% | Max outlier: {outliers_values.max():.2f}%")
        
        # 4. Recent returns and volatility (middle right)
        ax4 = plt.subplot(gs[1, 1])
        recent_returns = vol_analysis['recent_window_data']['returns']
        recent_vol = vol_analysis['recent_window_data']['rolling_volatility']
        
        recent_returns_clean = recent_returns[np.isfinite(recent_returns)]
        recent_vol_clean = recent_vol[np.isfinite(recent_vol)]
        
        if len(recent_returns_clean) > 0 and len(recent_vol_clean) > 0:
            ax4_returns = ax4
            ax4_vol = ax4_returns.twinx()
            
            dates_recent = recent_returns_clean.index
            returns_values_recent = recent_returns_clean.values * 100
            
            colors_recent = ['#27AE60' if x >= 0 else '#C0392B' for x in returns_values_recent]
            ax4_returns.bar(range(len(dates_recent)), returns_values_recent, 
                          color=colors_recent, alpha=0.7, width=0.8,
                          label='Daily returns (%)')
            
            vol_values_recent = recent_vol_clean.values * 100
            ax4_vol.plot(range(len(dates_recent)), vol_values_recent,
                        color='#8E44AD', linewidth=2.5, marker='s', markersize=4,
                        label=f'Volatility {window}d (%)', alpha=0.9)
            
            ax4_returns.set_xlabel('Date', fontweight='bold', fontsize=10)
            ax4_returns.set_ylabel('Returns (%)', fontweight='bold', fontsize=10, color='#27AE60')
            ax4_vol.set_ylabel(f'Volatility {window}d (%)', fontweight='bold', 
                              fontsize=10, color='#8E44AD')
            ax4_returns.tick_params(axis='y', labelcolor='#27AE60')
            ax4_vol.tick_params(axis='y', labelcolor='#8E44AD')
            
            tick_positions = range(0, len(dates_recent), max(1, len(dates_recent)//5))
            ax4_returns.set_xticks(tick_positions)
            ax4_returns.set_xticklabels([dates_recent[i].strftime('%d/%m') for i in tick_positions], 
                                       rotation=45, fontsize=8)
            
            ax4_returns.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax4_returns.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
            ax4_vol.grid(False)
            
            lines_returns, labels_returns = ax4_returns.get_legend_handles_labels()
            lines_vol, labels_vol = ax4_vol.get_legend_handles_labels()
            ax4_returns.legend(lines_returns + lines_vol, labels_returns + labels_vol,
                              loc='upper left', fontsize=8)
            
            ax4_returns.set_title(f'Recent Returns and Volatility\n({window} last days)', 
                                 fontweight='bold', fontsize=11)
        
        # 5. Historical returns and volatility WITHOUT OUTLIERS (bottom)
        ax5 = plt.subplot(gs[2, :])
        
        all_returns = vol_analysis['returns'].values * 100
        all_dates = vol_analysis['returns'].index
        
        # Filter outliers on returns
        returns_finite_mask = np.isfinite(all_returns)
        
        if np.sum(returns_finite_mask) > 0:
            # Detect outliers
            outliers_mask_full = filter_outliers(all_returns[returns_finite_mask], method='iqr', threshold=3)
            
            # Create complete mask for returns
            full_mask = np.zeros(len(all_returns), dtype=bool)
            full_mask[returns_finite_mask] = outliers_mask_full
            
            if np.sum(full_mask) > 0:
                all_dates_clean = all_dates[full_mask]
                all_returns_clean = all_returns[full_mask]
                
                # RECALCULATE volatility on cleaned returns
                returns_series_clean = pd.Series(all_returns_clean / 100, index=all_dates_clean)
                all_volatility_clean = (returns_series_clean.rolling(window=window).std() * 100).values
                
                ax5_returns = ax5
                ax5_vol = ax5_returns.twinx()
                
                x_range = range(len(all_dates_clean))
                
                # Returns with colored areas
                ax5_returns.fill_between(x_range, all_returns_clean, 0, 
                                        where=all_returns_clean >= 0, color='#27AE60', alpha=0.3,
                                        label='Positive returns')
                ax5_returns.fill_between(x_range, all_returns_clean, 0, 
                                        where=all_returns_clean < 0, color='#C0392B', alpha=0.3,
                                        label='Negative returns')
                ax5_returns.plot(x_range, all_returns_clean, 
                                color='#2C3E50', linewidth=0.8, alpha=0.7,
                                label='Daily returns (%)')
                
                # Rolling volatility (calculated on cleaned data)
                # Filter NaN from recalculated volatility
                vol_valid_mask = np.isfinite(all_volatility_clean)
                if np.sum(vol_valid_mask) > 0:
                    ax5_vol.plot(np.array(x_range)[vol_valid_mask], all_volatility_clean[vol_valid_mask],
                                color='#8E44AD', linewidth=2.5, alpha=0.9,
                                label=f'Rolling Volatility {window}d (%)')
                
                ax5_returns.set_xlabel('Period', fontweight='bold', fontsize=11)
                ax5_returns.set_ylabel('Returns (%)', fontweight='bold', fontsize=11, color='#2C3E50')
                ax5_vol.set_ylabel(f'Volatility {window}d (%)', fontweight='bold', 
                                  fontsize=11, color='#8E44AD')
                ax5_returns.tick_params(axis='y', labelcolor='#2C3E50')
                ax5_vol.tick_params(axis='y', labelcolor='#8E44AD')
                
                n_ticks = min(10, len(all_dates_clean))
                if n_ticks > 0:
                    tick_positions = np.linspace(0, len(all_dates_clean)-1, n_ticks, dtype=int)
                    ax5_returns.set_xticks(tick_positions)
                    ax5_returns.set_xticklabels([all_dates_clean[i].strftime('%m/%Y') for i in tick_positions], 
                                               rotation=45, fontsize=9)
                
                ax5_returns.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                ax5_returns.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
                ax5_vol.grid(False)
                
                lines_returns, labels_returns = ax5_returns.get_legend_handles_labels()
                lines_vol, labels_vol = ax5_vol.get_legend_handles_labels()
                ax5_returns.legend(lines_returns + lines_vol, labels_returns + labels_vol,
                                  loc='upper left', frameon=True, fancybox=True, 
                                  shadow=True, fontsize=9)
                
                n_outliers_removed = len(all_returns) - len(all_returns_clean)
                ax5_returns.set_title(f'Returns and Volatility - Complete History (Without outliers - {n_outliers_removed} excluded)', 
                                     fontweight='bold', fontsize=13, pad=15)
                
                # Adjust scales
                if len(all_returns_clean) > 0:
                    y_max_returns = max(abs(all_returns_clean)) * 1.2
                    ax5_returns.set_ylim(-y_max_returns, y_max_returns)
                
                if np.sum(vol_valid_mask) > 0:
                    vol_max = np.max(all_volatility_clean[vol_valid_mask]) * 1.2
                    if vol_max > 0:
                        ax5_vol.set_ylim(0, vol_max)
        
        plt.suptitle(f'Complete Analysis - Spread {market1}-{market2} ({contract}) - Volatility {window}d', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        print(f"\n📊 VOLATILITY & RETURNS ANALYSIS - {contract} (Volatility window: {window} days)")
        print("=" * 90)
        
        total_days = len(vol_analysis['returns'])
        valid_days = np.sum(np.isfinite(vol_analysis['returns']))
        print(f"📅 Analysis period: {valid_days}/{total_days} valid days")
        
        # OUTLIERS DIAGNOSTIC - IMPORTANT ADDITION
        print(f"\n🔍 DATA DIAGNOSTICS - BEFORE/AFTER FILTERING")
        print("=" * 50)
        
        # Raw data (with outliers)
        returns_raw = vol_analysis['returns'][np.isfinite(vol_analysis['returns'])]
        print(f"📊 Raw data: {len(returns_raw)} points")
        print(f"📊 Maximum gross yield: {returns_raw.max()*100:.2f}%")
        print(f"📊 Minimum gross yield: {returns_raw.min()*100:.2f}%")
        print(f"📊 Gross volatility: {returns_raw.std()*100:.2f}%")
        
        # Outlier detection on raw data
        returns_pct = returns_raw * 100
        outliers_mask = filter_outliers(returns_pct, method='iqr', threshold=3)
        returns_filtered = returns_pct[outliers_mask] / 100  # Back to decimal
        
        print(f"📊 Filtered data: {len(returns_filtered)} points")
        print(f"📊 Maximum filtered yield: {returns_filtered.max()*100:.2f}%")
        print(f"📊 Minimum filtered yield: {returns_filtered.min()*100:.2f}%")
        print(f"📊 Filtered volatility: {returns_filtered.std()*100:.2f}%")
        print(f"📊 Detected outliers: {len(returns_raw) - len(returns_filtered)}")
        
        # USE FILTERED DATA FOR STATISTICS
        returns_clean = pd.Series(returns_filtered)
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        # Recalculate statistics with filtered data
        if len(returns_clean) > 0:
            avg_return_clean = returns_clean.mean()
            return_volatility_clean = returns_clean.std()
        else:
            avg_return_clean = np.nan
            return_volatility_clean = np.nan
        
        if vol_analysis['current_volatility'] is not None and np.isfinite(vol_analysis['current_volatility']):
            print(f"📈 Current volatility ({window}d): {vol_analysis['current_volatility']*100:.2f}%")
        
        # DISPLAY STATISTICS WITH AND WITHOUT OUTLIERS
        print(f"\n📈 COMPARATIVE STATISTICS")
        print("=" * 40)
        
        if np.isfinite(vol_analysis['avg_return']):
            print(f"📊 Average yield (with outliers): {vol_analysis['avg_return']*100:.4f}%")
        
        
        if np.isfinite(vol_analysis['return_volatility']):
            print(f"📊 Total volatility (with outliers): {vol_analysis['return_volatility']*100:.4f}%")
        
        if np.isfinite(return_volatility_clean):
            print(f"📊 Total volatility (without outliers): {return_volatility_clean*100:.4f}%")
            
            if return_volatility_clean > 0 and np.isfinite(avg_return_clean):
                sharpe_ratio_daily = avg_return_clean / return_volatility_clean
                sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(252)
                
        
        if np.isfinite(vol_analysis['min_spread']) and np.isfinite(vol_analysis['max_spread']):
            print(f"📊 Spread min/max: {vol_analysis['min_spread']:+.3f} / {vol_analysis['max_spread']:+.3f} €/MWh")
            print(f"📊 Total amplitude: {vol_analysis['max_spread'] - vol_analysis['min_spread']:.3f} €/MWh")
        
        # Returns statistics WITH FILTERED DATA
        if len(returns_clean) > 0:
            print(f"📊 Positive days: {len(positive_returns)}/{len(returns_clean)} "
                  f"({len(positive_returns)/len(returns_clean)*100:.1f}%)")
            print(f"📊 Negative days: {len(negative_returns)}/{len(returns_clean)} "
                  f"({len(negative_returns)/len(returns_clean)*100:.1f}%)")
            
            if len(positive_returns) > 0:
                print(f"📊 Average yield (days+): {positive_returns.mean()*100:.4f}%")
            
            if len(negative_returns) > 0:
                print(f"📊 Average yield (days-): {negative_returns.mean()*100:.4f}%")
        
        # Average volatility
        vol_clean = vol_analysis['rolling_volatility'][np.isfinite(vol_analysis['rolling_volatility'])]
        if len(vol_clean) > 0:
            vol_mean = vol_clean.mean() * 100
            vol_max = vol_clean.max() * 100
            if np.isfinite(vol_mean) and np.isfinite(vol_max):
                print(f"📊 Volatility {window}d - Average: {vol_mean:.2f}% | Max: {vol_max:.2f}%")
        
        plt.show()


#================================= TIME SPREAD ==============================================================#


    def time_spread(self, market: str, type: str, contracts: Union[str, List[str]] = None,
                   periods: Union[str, List[str]] = None, graph: bool = True, 
                   volatility_window: Optional[int] = None):
        """
        Calculates time spreads between different contracts of the same market (ex: Nov25 vs Dec25)
        
        Args:
            market: Market to analyze ('TTF' or 'PEG') - REQUIRED
            type: Contract type ('standard' or 'composite') - REQUIRED
            contracts: Contract(s) to analyze (ex: 'Nov25 vs Dec25' or ['Nov25 vs Dec25', 'Dec25 vs Jan26'])
            periods: Period(s) for variations ('daily', 'weekly', 'monthly')
            graph: Display graph with moving averages (default: True)
            volatility_window: Window in days for volatility/return analysis (ex: 7, 30)
        
        Returns:
            dict: Time spread analysis results
        """
        if type.lower() not in ['standard', 'composite']:
            print("❌ Parameter 'type' must be 'standard' or 'composite'")
            return {}
        
        self._load_data()
        
        market = market.upper()
        
        # Define default contract pairs according to type
        if type.lower() == 'standard':
            default_contract_pairs = [
                'Nov25 vs Dec25', 'Dec25 vs Jan26', 'Jan26 vs Feb26', 'Feb26 vs Mar26',
                'Mar26 vs Apr26', 'Apr26 vs May26', 'May26 vs Jun26', 'Jun26 vs Jul26',
                'Jul26 vs Aug26', 'Aug26 vs Sep26', 'Sep26 vs Oct26', 'Oct26 vs Nov26'
            ]
        else:  # composite
            default_contract_pairs = [
                'Q425 vs Q126', 'Q126 vs Q226', 'Q226 vs Q326', 'Q326 vs Q426'
            ]
        
        if contracts is None:
            contracts_to_analyze = default_contract_pairs
        else:
            if isinstance(contracts, str):
                contracts_to_analyze = [contracts]
            else:
                contracts_to_analyze = contracts
        
        # Period parameters
        if periods is None:
            periods = ['daily', 'weekly', 'monthly']
        elif isinstance(periods, str):
            periods = [periods.lower()]
        else:
            periods = [p.lower() for p in periods]
        
        # Results
        results = {}
        
        print(f"\n📅 TIME SPREAD ANALYSIS - {market}")
        print("=" * 80)
        
        for contract_pair in contracts_to_analyze:
            print(f"\n📊 Contract pair: {contract_pair}")
            print("-" * 50)
            
            # Parse contract pair
            parsed_pair = self._parse_contract_pair(contract_pair)
            if not parsed_pair:
                print(f"❌ Invalid pair format: {contract_pair}")
                continue
                
            contract1, contract2 = parsed_pair
            
            # Check consistency with type
            if type.lower() == 'composite':
                if contract1 not in self.composite_contracts or contract2 not in self.composite_contracts:
                    print(f"❌ For type='composite', contracts must be composites")
                    print(f"   Available composite contracts: {list(self.composite_contracts.keys())}")
                    continue
            
            # Calculate current spread and variations
            spread_data = self._calculate_time_spread_analysis(
                market, contract1, contract2, type.lower(), periods
            )
            
            if not spread_data:
                print(f"❌ Insufficient data for {contract_pair}")
                continue
            
            results[contract_pair] = spread_data
            
            # Display results
            self._display_time_spread_results(contract_pair, spread_data, market)
            
            # Graph with moving averages if requested
            if graph:
                self._plot_time_spread_with_ma(
                    market, contract1, contract2, type.lower(), 
                    ma_periods=[50, 100]
                )
            
            # Volatility and returns analysis if requested
            if volatility_window is not None:
                vol_analysis = self._analyze_time_spread_volatility_and_returns(
                    market, contract1, contract2, type.lower(), volatility_window
                )
                if vol_analysis:
                    results[contract_pair]['volatility_analysis'] = vol_analysis
                    self._plot_time_spread_volatility_and_returns(vol_analysis, contract_pair, market, volatility_window)
        
        

    def _parse_contract_pair(self, contract_pair: str) -> Optional[Tuple[str, str]]:
        """
        Parse a contract string in format 'Nov25 vs Dec25' or 'Q126 vs Q226'
        """
        # Remove extra spaces
        contract_pair_clean = contract_pair.strip()
        
        # Look for 'vs' separator (case insensitive)
        separators = [' vs ', ' VS ', ' Vs ', ' vS ', '-', ' - ']
        
        for sep in separators:
            if sep in contract_pair_clean:
                parts = contract_pair_clean.split(sep)
                if len(parts) == 2:
                    contract1 = parts[0].strip()
                    contract2 = parts[1].strip()
                    return (contract1, contract2)
        
        return None

    def _calculate_time_spread_analysis(self, market: str, contract1: str, contract2: str,
                                      contract_type: str, periods: List[str]) -> Optional[Dict]:
        """
        Calculates current time spread and its variations over different periods
        """
        # Get current prices
        current_price1 = self._get_contract_price(market, contract1, contract_type)
        current_price2 = self._get_contract_price(market, contract2, contract_type)
        
        if not current_price1 or not current_price2:
            return None
        
        current_spread = current_price1['price'] - current_price2['price']
        current_date = current_price1['date']
        
        results = {
            'current_spread': current_spread,
            'current_date': current_date,
            'contract1_price': current_price1['price'],
            'contract2_price': current_price2['price'],
            'variations': {}
        }
        
        # Calculate variations for each period
        for period in periods:
            past_price1 = self._get_contract_price_at_period(
                market, contract1, contract_type, period
            )
            past_price2 = self._get_contract_price_at_period(
                market, contract2, contract_type, period
            )
            
            if past_price1 and past_price2:
                past_spread = past_price1['price'] - past_price2['price']
                variation_abs = current_spread - past_spread
                variation_pct = (variation_abs / abs(past_spread)) * 100 if past_spread != 0 else 0
                
                results['variations'][period] = {
                    'past_spread': past_spread,
                    'past_date': past_price1['date'],
                    'variation_abs': variation_abs,
                    'variation_pct': variation_pct
                }
        
        # RETURN RESULTS (this line was missing)
        return results
            

    def _display_time_spread_results(self, contract_pair: str, spread_data: Dict, market: str):
        """
        Displays time spread analysis results
        """
        current_spread = spread_data['current_spread']
        current_date = spread_data['current_date']
        
        print(f"💰 Current time spread: {current_spread:+.3f} €/MWh")
        print(f"   └─ {market} {contract_pair.split(' vs ')[0]}: {spread_data['contract1_price']:.3f} €/MWh")
        print(f"   └─ {market} {contract_pair.split(' vs ')[1]}: {spread_data['contract2_price']:.3f} €/MWh")
        print(f"📅 Date: {current_date.strftime('%d/%m/%Y')}")
        
        if spread_data['variations']:
            print("\n📊 Spread variations:")
            for period, var_data in spread_data['variations'].items():
                period_names = {'daily': 'Day', 'weekly': 'Week', 'monthly': 'Month'}
                period_name = period_names.get(period, period)
                
                color_circle = self._get_variation_color(var_data['variation_pct'])
                
                print(f"   • {period_name:<8}: {var_data['variation_abs']:+.3f} €/MWh "
                      f"{color_circle} {var_data['variation_pct']:+.1f}% "
                      f"(since {var_data['past_date'].strftime('%d/%m')})")

    def _plot_time_spread_with_ma(self, market: str, contract1: str, contract2: str,
                                contract_type: str, ma_periods: List[int] = [50, 200]):
        """
        Plot time spread graph with moving averages
        CORRECTED VERSION: Calculates complete historical spread for composites
        """
        # Get complete historical spread data
        spread_series = self._get_historical_time_spread(market, contract1, contract2, contract_type)
        
        if spread_series is None or len(spread_series) < max(ma_periods):
            print(f"⚠️  Insufficient historical data for MA{ma_periods} on {contract1} vs {contract2}")
            print(f"   (Available: {len(spread_series) if spread_series is not None else 0} days, required: {max(ma_periods)})")
            return
        
        # Calculate moving averages
        ma_data = {}
        for period in ma_periods:
            if len(spread_series) >= period:
                ma_data[f'MA{period}'] = spread_series.rolling(window=period).mean()
        
        # Create graph
        plt.figure(figsize=(14, 8))
        
        # Plot spread
        plt.plot(spread_series.index, spread_series.values, 
                color='#2E86AB', linewidth=1.5, label=f'Spread {contract1}-{contract2}', alpha=0.8)
        
        # Plot moving averages
        colors_ma = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (ma_name, ma_series) in enumerate(ma_data.items()):
            plt.plot(ma_series.index, ma_series.values, 
                    color=colors_ma[i % len(colors_ma)], linewidth=2, 
                    label=ma_name, alpha=0.9)
        
        # Graph configuration
        title_suffix = f"Composite {contract1} vs {contract2}" if contract_type == 'composite' else f"Standard {contract1} vs {contract2}"
        plt.title(f'Time Spread {market} - {title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Spread (€/MWh)', fontsize=12, fontweight='bold')
        
        # Zero line
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Date rotation
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        
        plt.show()

    def _get_historical_time_spread(self, market: str, contract1: str, contract2: str,
                                  contract_type: str) -> Optional[pd.Series]:
        """
        Obtains complete historical series of time spread
        """
        if contract_type == 'composite' and contract1 in self.composite_contracts and contract2 in self.composite_contracts:
            # For composites, calculate composite spread over entire history
            monthly_contracts1 = [f"{market}{m}" for m in self.composite_contracts[contract1]]
            monthly_contracts2 = [f"{market}{m}" for m in self.composite_contracts[contract2]]
            
            # Verify all contracts exist
            if not all(c in self._data for c in monthly_contracts1 + monthly_contracts2):
                return None
            
            # Find dates common to all contracts
            all_dates = None
            for contract_id in monthly_contracts1 + monthly_contracts2:
                df = self._data[contract_id].set_index('Date')
                if all_dates is None:
                    all_dates = set(df.index)
                else:
                    all_dates = all_dates.intersection(set(df.index))
            
            if not all_dates:
                return None
            
            all_dates = sorted(list(all_dates))
            
            # Calculate composite spread for each date
            spread_values = []
            for date in all_dates:
                price1 = sum(self._data[c].set_index('Date').loc[date, 'Close'] 
                           for c in monthly_contracts1) / len(monthly_contracts1)
                price2 = sum(self._data[c].set_index('Date').loc[date, 'Close'] 
                           for c in monthly_contracts2) / len(monthly_contracts2)
                spread_values.append(price1 - price2)
            
            return pd.Series(spread_values, index=all_dates)
        
        else:
            # For standard contracts
            contract_id1 = f"{market}{contract1}"
            contract_id2 = f"{market}{contract2}"
            
            if contract_id1 not in self._data or contract_id2 not in self._data:
                return None
            
            df1 = self._data[contract_id1].set_index('Date')
            df2 = self._data[contract_id2].set_index('Date')
            
            # Align dates
            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) == 0:
                return None
            
            df1_aligned = df1.loc[common_dates]
            df2_aligned = df2.loc[common_dates]
            
            return df1_aligned['Close'] - df2_aligned['Close']

    def _analyze_time_spread_volatility_and_returns(self, market: str, contract1: str, contract2: str, 
                                                  contract_type: str, window: int) -> Optional[Dict]:
        """
        Analyzes time spread volatility and returns on rolling window
        ONLY USES DATA UP TO D-1 (previous business day)
        """
        # Get complete historical spread series
        spread_series = self._get_historical_time_spread(market, contract1, contract2, contract_type)
        
        if spread_series is None or len(spread_series) < window + 1:
            return None
        
        # Sort by date and filter out today's data
        spread_series = spread_series.sort_index()
        today = pd.Timestamp.today().normalize()
        spread_series = spread_series[spread_series.index < today]  # ONLY DATA BEFORE TODAY
        
        if len(spread_series) < window + 1:
            return None
        
        # Calculate daily spread returns
        returns = spread_series.pct_change().dropna()
        
        # Calculate rolling volatility (NOT annualized)
        rolling_volatility = returns.rolling(window=window).std()  # Not annualized
        
        # Take last observations according to window for some analyses
        recent_data = spread_series.tail(window)
        recent_returns = returns.tail(window)
        recent_volatility = rolling_volatility.tail(window)
        
        return {
            'spread_values': spread_series,  # All data now
            'returns': returns,  # All returns
            'rolling_volatility': rolling_volatility,  # All volatility
            'current_volatility': recent_volatility.iloc[-1] if not recent_volatility.empty else None,
            'avg_return': returns.mean(),  # Average over entire period
            'return_volatility': returns.std(),  # Volatility over entire period
            'min_spread': spread_series.min(),
            'max_spread': spread_series.max(),
            'window': window,
            'recent_window_data': {  # Keep recent data separately
                'spread_values': recent_data,
                'returns': recent_returns,
                'rolling_volatility': recent_volatility
            }
        }

    def _plot_time_spread_volatility_and_returns(self, vol_analysis: Dict, contract_pair: str, 
                                               market: str, window: int):        
        """
        Plots time spread volatility and return graphs
        """
        # Helper function to detect and filter outliers
        def filter_outliers(data, method='iqr', threshold=3):
            """
            Filters outliers using the IQR or Z-score method
            Returns a Boolean mask for valid values
            """
            data_clean = data[np.isfinite(data)]
            if len(data_clean) == 0:
                return np.zeros(len(data), dtype=bool)
            
            if method == 'iqr':
                Q1 = np.percentile(data_clean, 25)
                Q3 = np.percentile(data_clean, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            else:  # z-score
                mean = np.mean(data_clean)
                std = np.std(data_clean)
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            
            return (data >= lower_bound) & (data <= upper_bound) & np.isfinite(data)
        
        # Create 2x2 layout
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
        
        # 1. Complete spread evolution (top left)
        ax1 = plt.subplot(gs[0, 0])
        spread_data = vol_analysis['spread_values']
        spread_clean_mask = np.isfinite(spread_data.values)
        
        if np.sum(spread_clean_mask) > 0:
            ax1.plot(spread_data.index[spread_clean_mask], spread_data.values[spread_clean_mask],
                    color='#2E86AB', linewidth=2, alpha=0.8)
            ax1.set_title(f'Time Spread Evolution {contract_pair}\n(Complete)', 
                         fontweight='bold', fontsize=12)
            ax1.set_ylabel('Spread (€/MWh)', fontweight='bold', fontsize=10)
            ax1.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            
            current_spread = spread_data.values[spread_clean_mask][-1]
            ax1.axhline(y=current_spread, color='red', linestyle=':', alpha=0.8, 
                       label=f'Current: {current_spread:.3f} €/MWh')
            ax1.legend(loc='best', fontsize=8)
        
        # 2. Recent spread evolution (top right)
        ax2 = plt.subplot(gs[0, 1])
        recent_data = vol_analysis['recent_window_data']['spread_values']
        recent_clean_mask = np.isfinite(recent_data.values)
        
        if np.sum(recent_clean_mask) > 0:
            ax2.plot(recent_data.index[recent_clean_mask], recent_data.values[recent_clean_mask],
                    color='#E74C3C', linewidth=2.5, marker='o', markersize=3)
            ax2.set_title(f'Recent Time Spread Evolution\n({window} last days)', 
                         fontweight='bold', fontsize=12)
            ax2.set_ylabel('Spread (€/MWh)', fontweight='bold', fontsize=10)
            ax2.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 3. Returns distribution WITHOUT OUTLIERS (middle left)
        ax3 = plt.subplot(gs[1, 0])
        returns_pct = vol_analysis['returns'].values * 100
        
        # Filter NaN/Inf
        returns_finite_mask = np.isfinite(returns_pct)
        returns_pct_finite = returns_pct[returns_finite_mask]
        
        # Detect and filter outliers
        if len(returns_pct_finite) > 0:
            outliers_mask = filter_outliers(returns_pct_finite, method='iqr', threshold=3)
            returns_no_outliers = returns_pct_finite[outliers_mask]
            n_outliers = len(returns_pct_finite) - len(returns_no_outliers)
            
            if len(returns_no_outliers) > 0:
                n, bins, patches = ax3.hist(returns_no_outliers, 
                                           bins=min(30, max(10, len(returns_no_outliers)//10)), 
                                           alpha=0.7, color='#4ECDC4', edgecolor='black', 
                                           density=True)
                
                # Color negative bars in red
                for patch, left_edge in zip(patches, bins):
                    if left_edge < 0:
                        patch.set_facecolor('#FF6B6B')
                        patch.set_alpha(0.7)
                
                ax3.set_title(f'Daily Returns Distribution\n(Without outliers - {n_outliers} excluded)', 
                             fontweight='bold', fontsize=12)
                ax3.set_xlabel('Return (%)', fontweight='bold', fontsize=10)
                ax3.set_ylabel('Density', fontweight='bold', fontsize=10)
                ax3.grid(True, which='major', axis='y', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
                
                # Statistics
                mean_return_no_outliers = np.mean(returns_no_outliers)
                ax3.axvline(x=mean_return_no_outliers, color='red', linestyle='--', linewidth=2,
                           label=f'Average: {mean_return_no_outliers:.3f}%')
                
                # Skewness and kurtosis
                from scipy import stats
                if len(returns_no_outliers) > 2:
                    skewness = stats.skew(returns_no_outliers)
                    kurtosis = stats.kurtosis(returns_no_outliers)
                    ax3.text(0.02, 0.98, 
                            f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}\nOutliers: {n_outliers}', 
                            transform=ax3.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)
                
                ax3.legend(fontsize=8)
        
        # 4. Recent returns and volatility (middle right)
        ax4 = plt.subplot(gs[1, 1])
        recent_returns = vol_analysis['recent_window_data']['returns']
        recent_vol = vol_analysis['recent_window_data']['rolling_volatility']
        
        recent_returns_clean = recent_returns[np.isfinite(recent_returns)]
        recent_vol_clean = recent_vol[np.isfinite(recent_vol)]
        
        if len(recent_returns_clean) > 0 and len(recent_vol_clean) > 0:
            ax4_returns = ax4
            ax4_vol = ax4_returns.twinx()
            
            dates_recent = recent_returns_clean.index
            returns_values_recent = recent_returns_clean.values * 100
            
            colors_recent = ['#27AE60' if x >= 0 else '#C0392B' for x in returns_values_recent]
            ax4_returns.bar(range(len(dates_recent)), returns_values_recent, 
                          color=colors_recent, alpha=0.7, width=0.8,
                          label='Daily returns (%)')
            
            vol_values_recent = recent_vol_clean.values * 100
            ax4_vol.plot(range(len(dates_recent)), vol_values_recent,
                        color='#8E44AD', linewidth=2.5, marker='s', markersize=4,
                        label=f'Volatility {window}d (%)', alpha=0.9)
            
            ax4_returns.set_xlabel('Date', fontweight='bold', fontsize=10)
            ax4_returns.set_ylabel('Returns (%)', fontweight='bold', fontsize=10, color='#27AE60')
            ax4_vol.set_ylabel(f'Volatility {window}d (%)', fontweight='bold', 
                              fontsize=10, color='#8E44AD')
            ax4_returns.tick_params(axis='y', labelcolor='#27AE60')
            ax4_vol.tick_params(axis='y', labelcolor='#8E44AD')
            
            tick_positions = range(0, len(dates_recent), max(1, len(dates_recent)//5))
            ax4_returns.set_xticks(tick_positions)
            ax4_returns.set_xticklabels([dates_recent[i].strftime('%d/%m') for i in tick_positions], 
                                       rotation=45, fontsize=8)
            
            ax4_returns.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax4_returns.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
            ax4_vol.grid(False)
            
            lines_returns, labels_returns = ax4_returns.get_legend_handles_labels()
            lines_vol, labels_vol = ax4_vol.get_legend_handles_labels()
            ax4_returns.legend(lines_returns + lines_vol, labels_returns + labels_vol,
                              loc='upper left', fontsize=8)
            
            ax4_returns.set_title(f'Recent Returns and Volatility\n({window} last days)', 
                                 fontweight='bold', fontsize=11)
        
        # 5. Historical returns and volatility WITHOUT OUTLIERS (bottom)
        ax5 = plt.subplot(gs[2, :])
        
        all_returns = vol_analysis['returns'].values * 100
        all_dates = vol_analysis['returns'].index
        
        # Filter outliers on returns
        returns_finite_mask = np.isfinite(all_returns)
        
        if np.sum(returns_finite_mask) > 0:
            # Detect outliers
            outliers_mask_full = filter_outliers(all_returns[returns_finite_mask], method='iqr', threshold=3)
            
            # Create complete mask for returns
            full_mask = np.zeros(len(all_returns), dtype=bool)
            full_mask[returns_finite_mask] = outliers_mask_full
            
            if np.sum(full_mask) > 0:
                all_dates_clean = all_dates[full_mask]
                all_returns_clean = all_returns[full_mask]
                
                # RECALCULATE volatility on cleaned returns
                returns_series_clean = pd.Series(all_returns_clean / 100, index=all_dates_clean)
                all_volatility_clean = (returns_series_clean.rolling(window=window).std() * 100).values
                
                ax5_returns = ax5
                ax5_vol = ax5_returns.twinx()
                
                x_range = range(len(all_dates_clean))
                
                # Returns with colored areas
                ax5_returns.fill_between(x_range, all_returns_clean, 0, 
                                        where=all_returns_clean >= 0, color='#27AE60', alpha=0.3,
                                        label='Positive returns')
                ax5_returns.fill_between(x_range, all_returns_clean, 0, 
                                        where=all_returns_clean < 0, color='#C0392B', alpha=0.3,
                                        label='Negative returns')
                ax5_returns.plot(x_range, all_returns_clean, 
                                color='#2C3E50', linewidth=0.8, alpha=0.7,
                                label='Daily returns (%)')
                
                # Rolling volatility (calculated on cleaned data)
                vol_valid_mask = np.isfinite(all_volatility_clean)
                if np.sum(vol_valid_mask) > 0:
                    ax5_vol.plot(np.array(x_range)[vol_valid_mask], all_volatility_clean[vol_valid_mask],
                                color='#8E44AD', linewidth=2.5, alpha=0.9,
                                label=f'Rolling Volatility {window}d (%)')
                
                ax5_returns.set_xlabel('Period', fontweight='bold', fontsize=11)
                ax5_returns.set_ylabel('Returns (%)', fontweight='bold', fontsize=11, color='#2C3E50')
                ax5_vol.set_ylabel(f'Volatility {window}d (%)', fontweight='bold', 
                                  fontsize=11, color='#8E44AD')
                ax5_returns.tick_params(axis='y', labelcolor='#2C3E50')
                ax5_vol.tick_params(axis='y', labelcolor='#8E44AD')
                
                n_ticks = min(10, len(all_dates_clean))
                if n_ticks > 0:
                    tick_positions = np.linspace(0, len(all_dates_clean)-1, n_ticks, dtype=int)
                    ax5_returns.set_xticks(tick_positions)
                    ax5_returns.set_xticklabels([all_dates_clean[i].strftime('%m/%Y') for i in tick_positions], 
                                               rotation=45, fontsize=9)
                
                ax5_returns.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                ax5_returns.grid(True, which='major', alpha=0.2, color='#888888', linestyle='-', linewidth=0.5)
                ax5_vol.grid(False)
                
                lines_returns, labels_returns = ax5_returns.get_legend_handles_labels()
                lines_vol, labels_vol = ax5_vol.get_legend_handles_labels()
                ax5_returns.legend(lines_returns + lines_vol, labels_returns + labels_vol,
                                  loc='upper left', frameon=True, fancybox=True, 
                                  shadow=True, fontsize=9)
                
                n_outliers_removed = len(all_returns) - len(all_returns_clean)
                ax5_returns.set_title(f'Returns and Volatility - Complete History (Without outliers - {n_outliers_removed} excluded)', 
                                     fontweight='bold', fontsize=13, pad=15)
        
        plt.suptitle(f'Complete Time Spread Analysis - {market} {contract_pair} - Volatility {window}d', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Display statistics
        print(f"\n📊 TIME SPREAD VOLATILITY & RETURNS ANALYSIS - {contract_pair} (Volatility window: {window} days)")
        print("=" * 90)
        
        total_days = len(vol_analysis['returns'])
        valid_days = np.sum(np.isfinite(vol_analysis['returns']))
        print(f"📅 Analysis period: {valid_days}/{total_days} valid days")
        
        # Statistics with filtered data
        returns_raw = vol_analysis['returns'][np.isfinite(vol_analysis['returns'])]
        returns_pct = returns_raw * 100
        outliers_mask = filter_outliers(returns_pct, method='iqr', threshold=3)
        returns_filtered = returns_pct[outliers_mask] / 100
        
        returns_clean = pd.Series(returns_filtered)
        
        if len(returns_clean) > 0:
            avg_return_clean = returns_clean.mean()
            return_volatility_clean = returns_clean.std()
            
            
            print(f"📊 Total volatility (without outliers): {return_volatility_clean*100:.4f}%")
            
            if return_volatility_clean > 0:
                sharpe_ratio_daily = avg_return_clean / return_volatility_clean
                sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(252)
                
        
        if np.isfinite(vol_analysis['min_spread']) and np.isfinite(vol_analysis['max_spread']):
            print(f"📊 Spread min/max: {vol_analysis['min_spread']:+.3f} / {vol_analysis['max_spread']:+.3f} €/MWh")
        
        plt.show()