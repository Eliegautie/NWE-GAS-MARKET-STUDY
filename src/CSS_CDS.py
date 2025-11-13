#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from IPython.display import display, HTML


class SpreadCalculator:
    """
    Clean Spark Spread and Clean Dark Spread Calculator
    """
    
    def __init__(self, base_path: str = r"C:\Users\Elie\Datas\TradingView"):
        self.data_path = base_path 
        self._setup_calculation_parameters()

        # Market paths definition
        self.market_paths = {
            'TTF': base_path + r"\CURVE\GAS\TTF",
            'COAL': base_path + r"\CURVE\COAL", 
            'POWER': base_path + r"\CURVE\POWER\GERMAN",
            'EUA': base_path + r"\CURVE\EUA"
        }
        
        # Month codes mapping for filename parsing
        self.month_codes = {
            'F': {'month': 1, 'name': 'Jan'},   'G': {'month': 2, 'name': 'Feb'},
            'H': {'month': 3, 'name': 'Mar'},   'J': {'month': 4, 'name': 'Apr'},
            'K': {'month': 5, 'name': 'May'},   'M': {'month': 6, 'name': 'Jun'},
            'N': {'month': 7, 'name': 'Jul'},   'Q': {'month': 8, 'name': 'Aug'},
            'U': {'month': 9, 'name': 'Sep'},   'V': {'month': 10, 'name': 'Oct'},
            'X': {'month': 11, 'name': 'Nov'},  'Z': {'month': 12, 'name': 'Dec'}
        }
        
        # Cache for loaded data
        self._data = {}
        self._common_date_info = {}

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse filename to extract contract information"""
        patterns = [
            (r'ICEENDEX_DLY_TFM([FGHJKMNQUVXZ])(\d{4})', 'TTF'),
            (r'ICEEUR_DLY_ATW([FGHJKMNQUVXZ])(\d{4})', 'COAL'),
            (r'ICEENDEX_DLY_GAB([FGHJKMNQUVXZ])(\d{4})', 'POWER'),
            (r'ICEENDEX_DLY_ECFZ(\d{4})', 'EUA'),
            (r'ICEEUR_DLY_ECFZ(\d{4})', 'EUA'),
            (r'ICEENDEX_DLY_EUA([FGHJKMNQUVXZ])(\d{4})', 'EUA'),
            (r'ICEEUR_DLY_EUA([FGHJKMNQUVXZ])(\d{4})', 'EUA'),
        ]
        
        for pattern, market in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if market == 'EUA' and 'ECFZ' in pattern:
                    year = groups[0]
                    month_info = self.month_codes['Z']
                else:
                    if len(groups) == 2:
                        month_code, year = groups
                        month_info = self.month_codes.get(month_code.upper())
                    else:
                        continue
                
                if month_info:
                    contract_name = f"{month_info['name']}{str(year)[-2:]}"
                    return {
                        'market': market,
                        'month': month_info['month'],
                        'month_name': month_info['name'],
                        'year': int(year),
                        'contract_id': f"{market}{contract_name}",
                        'contract_period': contract_name
                    }
        return None

    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column in DataFrame based on possible names (case insensitive)"""
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        return None

    def load_data(self):
        """Load all CSV data from market-specific directories using consistent pricing date"""
        if self._data:
            return
            
        all_dates = set()
        
        # First pass: find the most recent common date across all files
        for market, market_path in self.market_paths.items():
            if not os.path.exists(market_path):
                continue
                
            csv_files = glob.glob(os.path.join(market_path, "*.csv"))
            
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    date_col = self._find_column(df, ['time', 'date', 'timestamp'])
                    if date_col:
                        if df[date_col].dtype in ['int64', 'float64']:
                            dates = pd.to_datetime(df[date_col], unit='s')
                        else:
                            dates = pd.to_datetime(df[date_col])
                        all_dates.update(dates.dt.date)
                except Exception:
                    continue
        
        # Determine target date (most recent common date)
        target_date = max(all_dates) if all_dates else None
        
        # Second pass: load all data using consistent date
        for market, market_path in self.market_paths.items():
            if not os.path.exists(market_path):
                continue
                
            csv_files = glob.glob(os.path.join(market_path, "*.csv"))
            
            for file_path in csv_files:
                filename = os.path.basename(file_path)
                contract_info = self._parse_filename(filename)
                
                if contract_info:
                    try:
                        df = pd.read_csv(file_path)
                        
                        date_col = self._find_column(df, ['time', 'date', 'timestamp'])
                        close_col = self._find_column(df, ['close'])
                        
                        if date_col is None or close_col is None:
                            continue
                        
                        if df[date_col].dtype in ['int64', 'float64']:
                            df['Date'] = pd.to_datetime(df[date_col], unit='s')
                        else:
                            df['Date'] = pd.to_datetime(df[date_col])
                        
                        df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
                        df = df[df['Date'] >= '2025-01-01']
                        df = df.dropna(subset=['Close']).sort_values('Date')
                        
                        if len(df) == 0:
                            continue
                        
                        # Get price at target date
                        if target_date:
                            df_target = df[df['Date'].dt.date == target_date]
                            if len(df_target) == 0:
                                df_before = df[df['Date'].dt.date <= target_date]
                                if len(df_before) > 0:
                                    price_data = df_before.iloc[-1]
                                    price = price_data['Close']
                                    price_date = price_data['Date']
                                else:
                                    continue
                            else:
                                price_data = df_target.iloc[0]
                                price = price_data['Close']
                                price_date = price_data['Date']
                        else:
                            if len(df) >= 2:
                                price_data = df.iloc[-2]
                                price = price_data['Close']
                                price_date = price_data['Date']
                            else:
                                continue
                        
                        self._data[contract_info['contract_id']] = {
                            'data': df[['Date', 'Close']].copy(),
                            'info': contract_info,
                            'price': price,
                            'price_date': price_date
                        }
                        
                    except Exception:
                        continue

    def _setup_calculation_parameters(self):
        """Setup parameters for spread calculations"""
        self.composite_contracts = {
            'Q126': ['Jan26', 'Feb26', 'Mar26'],
            'Q226': ['Apr26', 'May26', 'Jun26'],
            'Q326': ['Jul26', 'Aug26', 'Sep26'],
            'Q426': ['Oct26', 'Nov26', 'Dec26'],
            'Q127': ['Jan27', 'Feb27', 'Mar27'],
            'Summer26': ['Apr26', 'May26', 'Jun26', 'Jul26', 'Aug26', 'Sep26'],
            'Winter26': ['Oct26', 'Nov26', 'Dec26','Jan27', 'Feb27', 'Mar27'],
            'Cal26': ['Jan26', 'Feb26', 'Mar26', 'Apr26', 'May26', 'Jun26',
                     'Jul26', 'Aug26', 'Sep26', 'Oct26', 'Nov26', 'Dec26'],
        }
        
        self.css_configs = [
            {'name': 'CSS_45', 'fuel_efficiency': 0.45, 'emissions_intensity': 0.1841},
            {'name': 'CSS_50', '29.}
        ]
        
        self.cds_configs = [
            {'name': 'CDS_35', 'fuel_efficiency': 0.35, 'emissions_intensity': 0.973, 'conversion': 6.978},
            {'name': 'CDS_45', 'fuel_efficiency': 0.45, 'emissions_intensity': 0.757, 'conversion': 6.978}
        ]
        
        self.eua_mapping = {
            2025: 'EUADec25',
            2026: 'EUADec26', 
            2027: 'EUADec27'
        }

    def _get_contract_year(self, contract_name: str) -> int:
        """Extract year from contract name"""
        year_str = contract_name[-2:]
        return 2000 + int(year_str)

    def _get_eua_contract_for_year(self, year: int) -> str:
        """Get appropriate EUA contract for a given year"""
        return self.eua_mapping.get(year, self.eua_mapping[max(self.eua_mapping.keys())])

    def _get_contracts_for_spread_calculation(self, contract_type: str, period: str) -> Dict[str, str]:
        """Get all necessary contracts for spread calculation"""
        contracts = {}
        
        if contract_type == 'standard':
            contracts['power'] = f"POWER{period}"
            contracts['fuel'] = f"TTF{period}"
            contracts['coal'] = f"COAL{period}"
            year = self._get_contract_year(period)
            contracts['eua'] = self._get_eua_contract_for_year(year)
            
        elif contract_type == 'composite':
            if period not in self.composite_contracts:
                raise ValueError(f"Unknown composite period: {period}")
            
            monthly_contracts = self.composite_contracts[period]
            power_contracts = [f"POWER{contract}" for contract in monthly_contracts]
            fuel_contracts = [f"TTF{contract}" for contract in monthly_contracts]
            coal_contracts = [f"COAL{contract}" for contract in monthly_contracts]
            first_contract_year = self._get_contract_year(monthly_contracts[0])
            eua_contract = self._get_eua_contract_for_year(first_contract_year)
            
            contracts = {
                'power_contracts': power_contracts,
                'fuel_contracts': fuel_contracts, 
                'coal_contracts': coal_contracts,
                'eua': eua_contract
            }
        
        return contracts

    def _find_reference_date(self, contract_ids: List[str]) -> pd.Timestamp:
        """Find the most recent common date for a list of contracts"""
        if not contract_ids:
            raise ValueError("No contract IDs provided")
        
        all_dates_by_contract = {}
        for contract_id in contract_ids:
            if contract_id in self._data:
                df = self._data[contract_id]['data']
                available_dates = set(df['Date'].dt.date)
                all_dates_by_contract[contract_id] = available_dates
        
        if not all_dates_by_contract:
            raise ValueError("No valid contracts found")
        
        common_dates = set.intersection(*all_dates_by_contract.values())
        
        if not common_dates:
            return self._find_best_available_date(contract_ids)
        
        sorted_dates = sorted(common_dates, reverse=True)
        today = pd.Timestamp.now().date()
        
        for date in sorted_dates:
            if date < today:
                return pd.Timestamp(date)
        
        return pd.Timestamp(sorted_dates[-1])

    def _find_best_available_date(self, contract_ids: List[str]) -> pd.Timestamp:
        """Find the best available date when no common date exists"""
        all_dates = set()
        date_coverage = {}
        
        for contract_id in contract_ids:
            if contract_id in self._data:
                df = self._data[contract_id]['data']
                contract_dates = set(df['Date'].dt.date)
                all_dates.update(contract_dates)
                
                for date in contract_dates:
                    date_coverage[date] = date_coverage.get(date, 0) + 1
        
        if not all_dates:
            raise ValueError("No dates available in any contract")
        
        max_coverage = max(date_coverage.values())
        best_dates = [date for date, coverage in date_coverage.items() if coverage == max_coverage]
        best_dates_sorted = sorted(best_dates, reverse=True)
        today = pd.Timestamp.now().date()
        
        for date in best_dates_sorted:
            if date < today:
                return pd.Timestamp(date)
        
        return pd.Timestamp(best_dates_sorted[-1])

    def _get_price_at_date(self, contract_id: str, target_date: pd.Timestamp) -> Optional[float]:
        """Get contract price at specific date"""
        if contract_id not in self._data:
            return None
        
        df = self._data[contract_id]['data']
        target_date_only = target_date.date()
        
        exact_match = df[df['Date'].dt.date == target_date_only]
        if not exact_match.empty:
            return exact_match['Close'].iloc[0]
        
        df_before = df[df['Date'].dt.date < target_date_only]
        if not df_before.empty:
            return df_before.iloc[-1]['Close']
        
        if not df.empty:
            return df.iloc[-1]['Close']
        
        return None

    def get_consistent_prices(self, contract_ids: List[str]) -> Dict[str, float]:
        """Get prices for multiple contracts using a consistent reference date"""
        reference_date = self._find_reference_date(contract_ids)
        
        prices = {}
        for contract_id in contract_ids:
            price = self._get_price_at_date(contract_id, reference_date)
            if price is not None:
                prices[contract_id] = price
        
        # Store common date info for display
        self._common_date_info['reference_date'] = reference_date
        self._common_date_info['contracts_used'] = list(prices.keys())
        
        return prices

    def _calculate_composite_price(self, contract_ids: List[str]) -> Optional[float]:
        """Calculate average price for a composite contract using consistent pricing"""
        if not contract_ids:
            return None
        
        prices = self.get_consistent_prices(contract_ids)
        if not prices:
            return None
            
        return np.mean(list(prices.values()))

    def calculate_clean_spark_spread(self, contract_type: str, period: str, config_index: int = 0) -> Optional[float]:
        """Calculate Clean Spark Spread"""
        try:
            contracts = self._get_contracts_for_spread_calculation(contract_type, period)
            config = self.css_configs[config_index]
            
            if contract_type == 'standard':
                contract_ids = [contracts['power'], contracts['fuel'], contracts['eua']]
                prices = self.get_consistent_prices(contract_ids)
                power_price = prices.get(contracts['power'])
                gas_price = prices.get(contracts['fuel'])
                eua_price = prices.get(contracts['eua'])
                
            elif contract_type == 'composite':
                power_price = self._calculate_composite_price(contracts['power_contracts'])
                gas_price = self._calculate_composite_price(contracts['fuel_contracts'])
                eua_price = self._get_price_at_date(contracts['eua'], pd.Timestamp.now() - pd.Timedelta(days=1))
            
            if None in [power_price, gas_price, eua_price]:
                return None
            
            fuel_efficiency = config['fuel_efficiency']
            emissions_intensity = config['emissions_intensity']
            css = power_price - (gas_price / fuel_efficiency + eua_price * (emissions_intensity / fuel_efficiency))
            
            return css
            
        except Exception:
            return None

    def calculate_clean_dark_spread(self, contract_type: str, period: str, config_index: int = 0) -> Optional[float]:
        """Calculate Clean Dark Spread"""
        try:
            contracts = self._get_contracts_for_spread_calculation(contract_type, period)
            config = self.cds_configs[config_index]
            
            if contract_type == 'standard':
                contract_ids = [contracts['power'], contracts['coal'], contracts['eua']]
                prices = self.get_consistent_prices(contract_ids)
                power_price = prices.get(contracts['power'])
                coal_price = prices.get(contracts['coal'])
                eua_price = prices.get(contracts['eua'])
                
            elif contract_type == 'composite':
                power_price = self._calculate_composite_price(contracts['power_contracts'])
                coal_price = self._calculate_composite_price(contracts['coal_contracts'])
                eua_price = self._get_price_at_date(contracts['eua'], pd.Timestamp.now() - pd.Timedelta(days=1))
            
            if None in [power_price, coal_price, eua_price]:
                return None
            
            fuel_efficiency = config['fuel_efficiency']
            emissions_intensity = config['emissions_intensity']
            conversion = config['conversion']
            cds = power_price - ((coal_price / conversion) / fuel_efficiency + eua_price * emissions_intensity)
            
            return cds
            
        except Exception:
            return None

    def get_available_composite_periods(self) -> List[str]:
        """Return only composite periods with recent available data"""
        available_periods = []
        
        possible_composites = ['Q126', 'Q226', 'Q326', 'Summer26', 'Q426', 'Cal26', 'Q127', 'Winter26']
        
        for period in possible_composites:
            try:
                contracts = self._get_contracts_for_spread_calculation('composite', period)
                power_available = [c for c in contracts['power_contracts'] if c in self._data]
                ttf_available = [c for c in contracts['fuel_contracts'] if c in self._data]
                coal_available = [c for c in contracts['coal_contracts'] if c in self._data]
                eua_available = contracts['eua'] in self._data
                
                if (len(power_available) >= 2 and len(ttf_available) >= 2 and 
                    len(coal_available) >= 2 and eua_available):
                    available_periods.append(period)
                    
            except Exception:
                continue
                
        return available_periods

    def get_available_standard_periods(self) -> List[str]:
        """Return only standard periods with recent available data in chronological order"""
        available_periods = []
        
        chronological_order = [            
            'Dec25', 'Jan26', 'Feb26', 'Mar26', 'Apr26', 'May26', 'Jun26', 
            'Jul26', 'Aug26', 'Sep26', 'Oct26', 'Nov26', 'Dec26',
            'Jan27', 'Feb27', 'Mar27'
        ]
        
        for period in chronological_order:
            try:
                contracts = self._get_contracts_for_spread_calculation('standard', period)
                power_available = contracts['power'] in self._data
                ttf_available = contracts['fuel'] in self._data
                coal_available = contracts['coal'] in self._data
                eua_available = contracts['eua'] in self._data
                
                if power_available and ttf_available and coal_available and eua_available:
                    available_periods.append(period)
                    
            except Exception:
                continue
        
        return available_periods

    def generate_current_spreads_table(self, contract_type: str = "composite") -> pd.DataFrame:
        """Generate current spreads table for specified contract type"""
        if contract_type.lower() == "composite":
            available_periods = self.get_available_composite_periods()
        else:
            available_periods = self.get_available_standard_periods()
        
        if not available_periods:
            return pd.DataFrame()
        
        table_data = []
        
        for period in available_periods:
            row_data = {'Contract': period}
            
            # Calculate all spread configurations
            css_50 = self.calculate_clean_spark_spread(contract_type, period, 1)
            css_45 = self.calculate_clean_spark_spread(contract_type, period, 0)
            cds_45 = self.calculate_clean_dark_spread(contract_type, period, 1)
            cds_35 = self.calculate_clean_dark_spread(contract_type, period, 0)
            
            row_data['CSS_50'] = css_50
            row_data['CSS_45'] = css_45
            row_data['CDS_45'] = cds_45
            row_data['CDS_35'] = cds_35
            
            table_data.append(row_data)
        
        if table_data:
            df = pd.DataFrame(table_data)
            column_order = ['Contract', 'CSS_50', 'CSS_45', 'CDS_45', 'CDS_35']
            return df[[col for col in column_order if col in df.columns]]
        else:
            return pd.DataFrame()

    def display_excel_format_with_merged_headers(self, df: pd.DataFrame, title: str = "Spreads Analysis"):
        """Display table with merged headers in Excel format"""
        if df.empty:
            return
        
        styled_df = df.copy().set_index('Contract')
        
        for col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else ""
            )
        
        html_output = f"""
        <div style="margin: 20px 0;">
            <h3 style="text-align: center; color: #343a40;">🎯 {title}</h3>
            <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; margin: 0 auto; font-family: Arial, sans-serif; font-size: 12px;">
                <thead>
                    <tr style="background-color: #343a40; color: white;">
                        <th style="border: 1px solid #454d55; padding: 10px;"></th>
                        <th colspan="2" style="border: 1px solid #454d55; padding: 10px; text-align: center;">CSS</th>
                        <th colspan="2" style="border: 1px solid #454d55; padding: 10px; text-align: center;">CDS</th>
                    </tr>
                    <tr style="background-color: #495057; color: white;">
                        <th style="border: 1px solid #454d55; padding: 8px; text-align: center;">Contract</th>
                        <th style="border: 1px solid #454d55; padding: 8px; text-align: center;">0.50</th>
                        <th style="border: 1px solid #454d55; padding: 8px; text-align: center;">0.45</th>
                        <th style="border: 1px solid #454d55; padding: 8px; text-align: center;">0.45</th>
                        <th style="border: 1px solid #454d55; padding: 8px; text-align: center;">0.35</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for contract in styled_df.index:
            html_output += f'<tr><td style="border: 1px solid #dee2e6; padding: 6px 8px; font-weight: bold; text-align: left;">{contract}</td>'
            
            for col in styled_df.columns:
                value = styled_df.loc[contract, col]
                if value:
                    try:
                        num_val = float(value)
                        if num_val > 0:
                            color_style = 'background-color: #d4edda; color: #155724; font-weight: bold;'
                        elif num_val < 0:
                            color_style = 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                        else:
                            color_style = 'background-color: #fff3cd; color: #856404;'
                    except:
                        color_style = ''
                else:
                    color_style = 'background-color: #f8f9fa; color: #6c757d;'
                
                html_output += f'<td style="border: 1px solid #dee2e6; padding: 6px 8px; text-align: center; {color_style}">{value}</td>'
            
            html_output += '</tr>'
        
        html_output += """
                </tbody>
            </table>
        </div>
        """
        
        display(HTML(html_output))

    def _get_price_at_date_simple(self, contract_id: str, target_date: pd.Timestamp) -> Optional[float]:
        """Get contract price at specific date using exact match"""
        if contract_id not in self._data:
            return None
        
        df = self._data[contract_id]['data']
        target_date = pd.Timestamp(target_date)
        
        exact_match = df[df['Date'].dt.normalize() == target_date.normalize()]
        if not exact_match.empty:
            return exact_match['Close'].iloc[0]
        
        df_before = df[df['Date'].dt.normalize() < target_date.normalize()]
        if not df_before.empty:
            return df_before.iloc[-1]['Close']
        
        return None

    def track_spreads_evolution(self, contract_type: str = "composite", lookback_days: int = 30):
        """Track CSS and CDS spreads evolution over a period"""
        if contract_type.lower() == "composite":
            contract_periods = ['Q126', 'Q226', 'Q326', 'Summer26', 'Q426', 'Cal26', 'Q127', 'Winter26']
        else:
            contract_periods = ['Dec25', 'Jan26', 'Feb26', 'Mar26', 'Apr26', 'May26', 'Jun26', 'Jul26', 'Aug26', 'Sep26', 'Oct26', 'Sep26', 'Nov26', 'Dec26', 'Jan27', 'Feb27', 'Mar27']
        
        # Store evolution period info
        self._common_date_info['evolution_period'] = f"{lookback_days} days"
        self._common_date_info['evolution_contracts'] = contract_periods
        
        all_contracts_needed = []
        for period in contract_periods:
            contracts = self._get_contracts_for_spread_calculation(contract_type, period)
            if contract_type == 'standard':
                all_contracts_needed.extend([contracts['power'], contracts['fuel'], contracts['coal'], contracts['eua']])
            else:
                all_contracts_needed.extend(contracts['power_contracts'] + contracts['fuel_contracts'] + contracts['coal_contracts'] + [contracts['eua']])
        
        common_dates = None
        for contract_id in all_contracts_needed:
            if contract_id in self._data:
                df = self._data[contract_id]['data']
                contract_dates = set(df['Date'].dt.date)
                if common_dates is None:
                    common_dates = contract_dates
                else:
                    common_dates = common_dates.intersection(contract_dates)
        
        if not common_dates:
            return {}
        
        sorted_dates = sorted(common_dates, reverse=True)
        today = pd.Timestamp.now().date()
        end_date = None
        
        for date in sorted_dates:
            if date < today:
                end_date = pd.Timestamp(date)
                break
        
        if end_date is None:
            end_date = pd.Timestamp(sorted_dates[-1])
        
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        evolution_data = {}
        
        for period in contract_periods:
            try:
                contracts = self._get_contracts_for_spread_calculation(contract_type, period)
                
                if contract_type == 'standard':
                    reference_contract = contracts['power']
                else:
                    reference_contract = contracts['power_contracts'][0] if contracts['power_contracts'] else None
                
                if not reference_contract or reference_contract not in self._data:
                    evolution_data[period] = None
                    continue
                
                df = self._data[reference_contract]['data']
                available_dates = df[
                    (df['Date'].dt.normalize() >= start_date) & 
                    (df['Date'].dt.normalize() <= end_date)
                ]['Date'].sort_values()
                
                if len(available_dates) == 0:
                    evolution_data[period] = None
                    continue
                
                dates = []
                css_45_values = []
                css_50_values = []
                cds_35_values = []
                cds_45_values = []
                
                for trading_date in available_dates:
                    try:
                        if contract_type == 'standard':
                            power_price = self._get_price_at_date_simple(contracts['power'], trading_date)
                            gas_price = self._get_price_at_date_simple(contracts['fuel'], trading_date)
                            coal_price = self._get_price_at_date_simple(contracts['coal'], trading_date)
                            eua_price = self._get_price_at_date_simple(contracts['eua'], trading_date)
                        else:
                            power_prices = []
                            for contract in contracts['power_contracts']:
                                price = self._get_price_at_date_simple(contract, trading_date)
                                if price is not None:
                                    power_prices.append(price)
                            
                            gas_prices = []
                            for contract in contracts['fuel_contracts']:
                                price = self._get_price_at_date_simple(contract, trading_date)
                                if price is not None:
                                    gas_prices.append(price)
                            
                            coal_prices = []
                            for contract in contracts['coal_contracts']:
                                price = self._get_price_at_date_simple(contract, trading_date)
                                if price is not None:
                                    coal_prices.append(price)
                            
                            eua_price = self._get_price_at_date_simple(contracts['eua'], trading_date)
                            
                            power_price = np.mean(power_prices) if power_prices else None
                            gas_price = np.mean(gas_prices) if gas_prices else None
                            coal_price = np.mean(coal_prices) if coal_prices else None
                        
                        if all(x is not None for x in [power_price, gas_price, coal_price, eua_price]):
                            css_45 = power_price - (gas_price / 0.45 + eua_price * (0.1841 / 0.45))
                            css_50 = power_price - (gas_price / 0.5 + eua_price * (0.1841 / 0.5))
                            cds_35 = power_price - ((coal_price / 6.978) / 0.35 + eua_price * 0.973)
                            cds_45 = power_price - ((coal_price / 6.978) / 0.45 + eua_price * 0.757)
                            
                            dates.append(trading_date)
                            css_45_values.append(css_45)
                            css_50_values.append(css_50)
                            cds_35_values.append(cds_35)
                            cds_45_values.append(cds_45)
                        
                    except Exception:
                        continue
                
                if dates:
                    evolution_data[period] = {
                        'dates': dates,
                        'CSS_45': css_45_values,
                        'CSS_50': css_50_values,
                        'CDS_35': cds_35_values,
                        'CDS_45': cds_45_values
                    }
                else:
                    evolution_data[period] = None
                
            except Exception:
                evolution_data[period] = None
        
        return evolution_data

    def display_spreads_evolution_table(self, evolution_data: Dict, title: str = "Spreads Evolution"):
        """Display spreads evolution table with variations"""
        valid_contracts = {k: v for k, v in evolution_data.items() if v is not None and len(v['dates']) > 0}
        
        if not valid_contracts:
            return None
        
        table_data = []
        
        for contract, data in valid_contracts.items():
            dates = data['dates']
            css_45 = data['CSS_45']
            css_50 = data['CSS_50']
            cds_35 = data['CDS_35']
            cds_45 = data['CDS_45']
            
            if len(css_45) >= 2:
                css_45_start, css_45_end = css_45[0], css_45[-1]
                css_50_start, css_50_end = css_50[0], css_50[-1]
                cds_35_start, cds_35_end = cds_35[0], cds_35[-1]
                cds_45_start, cds_45_end = cds_45[0], cds_45[-1]
                
                css_45_var = ((css_45_end - css_45_start) / abs(css_45_start)) * 100 if css_45_start != 0 else 0
                css_50_var = ((css_50_end - css_50_start) / abs(css_50_start)) * 100 if css_50_start != 0 else 0
                cds_35_var = ((cds_35_end - cds_35_start) / abs(cds_35_start)) * 100 if cds_35_start != 0 else 0
                cds_45_var = ((cds_45_end - cds_45_start) / abs(cds_45_start)) * 100 if cds_45_start != 0 else 0
                
                row_data = {
                    'Contract': contract,
                    'CSS_45_Price': f"{css_45_end:.2f}",
                    'CSS_45_Var': f"{css_45_var:+.1f}%",
                    'CSS_50_Price': f"{css_50_end:.2f}",
                    'CSS_50_Var': f"{css_50_var:+.1f}%",
                    'CDS_35_Price': f"{cds_35_end:.2f}",
                    'CDS_35_Var': f"{cds_35_var:+.1f}%",
                    'CDS_45_Price': f"{cds_45_end:.2f}",
                    'CDS_45_Var': f"{cds_45_var:+.1f}%"
                }
                
                table_data.append(row_data)
        
        if not table_data:
            return None
        
        df = pd.DataFrame(table_data)
        
        html_output = f"""
        <div style="margin: 20px 0;">
            <h3 style="text-align: center; color: #343a40;">📈 {title}</h3>
            <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; margin: 0 auto; font-family: Arial, sans-serif; font-size: 10px;">
                <thead>
                    <tr style="background-color: #343a40; color: white;">
                        <th rowspan="2" style="border: 1px solid #454d55; padding: 8px; text-align: center;">Contract</th>
                        <th colspan="8" style="border: 1px solid #454d55; padding: 8px; text-align: center;">Spreads Evolution (€/MWh)</th>
                    </tr>
                    <tr style="background-color: #495057; color: white;">
                        <th colspan="2" style="border: 1px solid #454d55; padding: 6px; text-align: center;">CSS 45%</th>
                        <th colspan="2" style="border: 1px solid #454d55; padding: 6px; text-align: center;">CSS 50%</th>
                        <th colspan="2" style="border: 1px solid #454d55; padding: 6px; text-align: center;">CDS 35%</th>
                        <th colspan="2" style="border: 1px solid #454d55; padding: 6px; text-align: center;">CDS 45%</th>
                    </tr>
                    <tr style="background-color: #5a6268; color: white;">
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;"></th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Price</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Var %</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Price</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Var %</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Price</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Var %</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Price</th>
                        <th style="border: 1px solid #454d55; padding: 4px; text-align: center;">Var %</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in df.iterrows():
            html_output += f'<tr><td style="border: 1px solid #dee2e6; padding: 6px; font-weight: bold; text-align: left;">{row["Contract"]}</td>'
            
            for var_col in ['CSS_45_Var', 'CSS_50_Var', 'CDS_35_Var', 'CDS_45_Var']:
                price_col = var_col.replace('_Var', '_Price')
                html_output += f'<td style="border: 1px solid #dee2e6; padding: 6px; text-align: center;">{row[price_col]}</td>'
                
                var_value = float(row[var_col].replace('%', '').replace('+', ''))
                color_style = 'color: #155724; background-color: #d4edda;' if var_value > 0 else 'color: #721c24; background-color: #f8d7da;' if var_value < 0 else ''
                html_output += f'<td style="border: 1px solid #dee2e6; padding: 6px; text-align: center; {color_style}">{row[var_col]}</td>'
            
            html_output += '</tr>'
        
        html_output += "</tbody></table></div>"
        display(HTML(html_output))
        
        return df

    def display_analysis_info(self, analysis_type: str = "General Analysis"):
        
        """
        Display analysis information including common dates and periods used
        """
        info_html = f"""
        <div style="margin: 10px 0; padding: 12px; background-color: #f8f9fa; border-left: 4px solid #007bff; border-radius: 4px;">
            <h5 style="color: #343a40; margin-bottom: 8px;">📊 Analysis Information - {analysis_type}</h5>
        """
        
        if 'reference_date' in self._common_date_info:
            info_html += f"""
            <p style="margin: 3px 0; font-size: 13px;"><strong>Common Pricing Date:</strong> {self._common_date_info['reference_date'].strftime('%Y-%m-%d')}</p>
            """
        
        if 'evolution_period' in self._common_date_info and "Evolution" in analysis_type:
            info_html += f"""
            <p style="margin: 3px 0; font-size: 13px;"><strong>Analysis Period:</strong> Last {self._common_date_info['evolution_period']}</p>
            """
            
            if 'evolution_contracts' in self._common_date_info:
                contracts_str = ', '.join(self._common_date_info['evolution_contracts'])
                info_html += f"""
                <p style="margin: 3px 0; font-size: 13px;"><strong>Contracts Analyzed:</strong> {contracts_str}</p>
                """
        
        # Add information about available contracts for current spreads
        if "Current Spreads" in analysis_type:
            if "Composite" in analysis_type:
                available_periods = self.get_available_composite_periods()
                if available_periods:
                    periods_str = ', '.join(available_periods)
                    info_html += f"""
                    <p style="margin: 3px 0; font-size: 13px;"><strong>Available Composite Periods:</strong> {periods_str}</p>
                    """
            elif "Standard" in analysis_type:
                available_periods = self.get_available_standard_periods()
                if available_periods:
                    periods_str = ', '.join(available_periods)
                    info_html += f"""
                    <p style="margin: 3px 0; font-size: 13px;"><strong>Available Standard Periods:</strong> {periods_str}</p>
                    """
        
        # Add data quality information
        if 'contracts_used' in self._common_date_info:
            contracts_count = len(self._common_date_info['contracts_used'])
            info_html += f"""
            <p style="margin: 3px 0; font-size: 13px;"><strong>Contracts Used for Pricing:</strong> {contracts_count} contracts</p>
            """
        
        info_html += "</div>"
        display(HTML(info_html))

    def generate_complete_analysis(self, lookback_days: int = 30):
        """
        Generate complete analysis with current spreads and evolution
        Returns: Dictionary with all results and dataframes
        """
        print("🎯 GENERATING COMPLETE SPREADS ANALYSIS")
        print("=" * 50)
        
        # Load data first
        self.load_data()
        
        results = {}
        
        # 1. Current Spreads - Composite Contracts
        print("\n1. CURRENT SPREADS - COMPOSITE CONTRACTS")
        composite_df = self.generate_current_spreads_table("composite")
        if not composite_df.empty:
            self.display_excel_format_with_merged_headers(composite_df, "Current Spreads - Composite Contracts")
            self.display_analysis_info("Composite Contracts - Current Spreads")
            results['composite_current'] = composite_df
        else:
            print("No data available for composite contracts")
        
        # 2. Current Spreads - Standard Contracts  
        print("\n2. CURRENT SPREADS - STANDARD CONTRACTS")
        standard_df = self.generate_current_spreads_table("standard")
        if not standard_df.empty:
            self.display_excel_format_with_merged_headers(standard_df, "Current Spreads - Standard Contracts")
            self.display_analysis_info("Standard Contracts - Current Spreads")
            results['standard_current'] = standard_df
        else:
            print("No data available for standard contracts")
        
        # 3. Spreads Evolution - Composite Contracts
        print("\n3. SPREADS EVOLUTION - COMPOSITE CONTRACTS")
        evolution_composite = self.track_spreads_evolution("composite", lookback_days)
        evolution_composite_df = self.display_spreads_evolution_table(evolution_composite, "Spreads Evolution - Composite Contracts")
        if evolution_composite_df is not None:
            self.display_analysis_info("Composite Contracts - Evolution")
            results['composite_evolution'] = evolution_composite_df
        
        # 4. Spreads Evolution - Standard Contracts
        print("\n4. SPREADS EVOLUTION - STANDARD CONTRACTS")
        evolution_standard = self.track_spreads_evolution("standard", lookback_days)
        evolution_standard_df = self.display_spreads_evolution_table(evolution_standard, "Spreads Evolution - Standard Contracts")
        if evolution_standard_df is not None:
            self.display_analysis_info("Standard Contracts - Evolution")
            results['standard_evolution'] = evolution_standard_df
        
        return results

