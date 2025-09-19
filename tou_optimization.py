import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class TOUOptimizer:
    """Class for TOU optimization simulations including load shifting and storage"""
    
    def __init__(self, baseline_profile, pricing_model, tou_periods):
        self.baseline_profile = baseline_profile
        self.pricing_model = pricing_model
        self.tou_periods = tou_periods
        self.hourly_rates = self._calculate_hourly_rates()
    
    def _calculate_hourly_rates(self):
        """Calculate hourly rates based on pricing model and TOU periods"""
        rates = np.zeros(24)
        
        if 'rate' in self.pricing_model:  # Flat rate
            rates[:] = self.pricing_model['rate']
        else:
            # TOU pricing
            for hour in range(24):
                if 'peak_hours' in self.pricing_model and hour in self.pricing_model['peak_hours']:
                    rates[hour] = self.pricing_model['peak_rate']
                elif 'shoulder_hours' in self.pricing_model and hour in self.pricing_model['shoulder_hours']:
                    rates[hour] = self.pricing_model['shoulder_rate']
                elif 'off_peak_hours' in self.pricing_model and hour in self.pricing_model['off_peak_hours']:
                    rates[hour] = self.pricing_model['off_peak_rate']
                else:
                    rates[hour] = self.pricing_model.get('off_peak_rate', self.pricing_model.get('peak_rate', 0.15))
        
        return rates
    
    def calculate_baseline_cost(self):
        """Calculate baseline cost without optimization"""
        return np.sum(self.baseline_profile * self.hourly_rates)
    
    def simulate_load_shifting(self, shiftable_percentage=0.3, flexibility_window=6):
        """Simulate load shifting optimization"""
        baseline_cost = self.calculate_baseline_cost()
        
        # Identify shiftable load (assume percentage of total load can be shifted)
        shiftable_load = self.baseline_profile * shiftable_percentage
        fixed_load = self.baseline_profile * (1 - shiftable_percentage)
        
        # Find optimal load distribution
        optimized_profile = self._optimize_load_shifting(
            fixed_load, shiftable_load, flexibility_window
        )
        
        optimized_cost = np.sum(optimized_profile * self.hourly_rates)
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost) * 100
        
        return {
            'baseline_profile': self.baseline_profile,
            'optimized_profile': optimized_profile,
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'savings': savings,
            'savings_pct': savings_pct,
            'shiftable_percentage': shiftable_percentage,
            'method': 'load_shifting'
        }
    
    def _optimize_load_shifting(self, fixed_load, shiftable_load, window):
        """Optimize load shifting using simple heuristic"""
        optimized_profile = fixed_load.copy()
        total_shiftable = np.sum(shiftable_load)
        
        # Sort hours by rate (cheapest first)
        rate_order = np.argsort(self.hourly_rates)
        
        # Distribute shiftable load to cheapest hours first
        remaining_load = total_shiftable
        for hour in rate_order:
            if remaining_load <= 0:
                break
            
            # Add some load to this hour (up to original + some flexibility)
            max_additional = shiftable_load[hour] * 2  # Can double the original shiftable load
            additional = min(remaining_load, max_additional)
            optimized_profile[hour] += additional
            remaining_load -= additional
        
        return optimized_profile
    
    def simulate_battery_storage(self, battery_capacity_kwh=10, battery_power_kw=5, efficiency=0.9):
        """Simulate battery storage optimization"""
        baseline_cost = self.calculate_baseline_cost()
        
        # Simple battery optimization: charge during cheap hours, discharge during expensive
        optimized_profile, battery_schedule = self._optimize_battery_usage(
            self.baseline_profile, battery_capacity_kwh, battery_power_kw, efficiency
        )
        
        optimized_cost = np.sum(optimized_profile * self.hourly_rates)
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost) * 100
        
        return {
            'baseline_profile': self.baseline_profile,
            'optimized_profile': optimized_profile,
            'battery_schedule': battery_schedule,
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'savings': savings,
            'savings_pct': savings_pct,
            'battery_capacity': battery_capacity_kwh,
            'battery_power': battery_power_kw,
            'method': 'battery_storage'
        }
    
    def _optimize_battery_usage(self, load_profile, capacity, power, efficiency):
        """Simple battery optimization algorithm"""
        optimized_profile = load_profile.copy()
        battery_schedule = np.zeros(24)  # Positive = charging, negative = discharging
        battery_soc = np.zeros(25)  # State of charge (start with 0)
        
        # Create rate ranking
        rate_ranking = np.argsort(self.hourly_rates)
        cheap_hours = rate_ranking[:8]  # 8 cheapest hours
        expensive_hours = rate_ranking[-8:]  # 8 most expensive hours
        
        # Simple strategy: charge during cheap hours, discharge during expensive hours
        for hour in range(24):
            if hour in cheap_hours and battery_soc[hour] < capacity:
                # Charge battery
                charge_power = min(power, capacity - battery_soc[hour])
                battery_schedule[hour] = charge_power
                optimized_profile[hour] += charge_power
                battery_soc[hour + 1] = battery_soc[hour] + charge_power * efficiency
            
            elif hour in expensive_hours and battery_soc[hour] > 0:
                # Discharge battery
                discharge_power = min(power, battery_soc[hour], load_profile[hour])
                battery_schedule[hour] = -discharge_power
                optimized_profile[hour] -= discharge_power
                battery_soc[hour + 1] = battery_soc[hour] - discharge_power
            
            else:
                battery_soc[hour + 1] = battery_soc[hour]
        
        return optimized_profile, battery_schedule
    
    def simulate_demand_response(self, dr_events=None, reduction_percentage=0.2):
        """Simulate demand response events"""
        if dr_events is None:
            # Default: DR events during peak hours
            dr_events = [h for h in range(24) if h in self.pricing_model.get('peak_hours', [17, 18, 19, 20, 21])]
        
        baseline_cost = self.calculate_baseline_cost()
        
        # Apply demand reduction during DR events
        optimized_profile = self.baseline_profile.copy()
        dr_savings = 0
        
        for hour in dr_events:
            reduction = optimized_profile[hour] * reduction_percentage
            optimized_profile[hour] -= reduction
            # Assume some incentive payment for DR participation
            dr_savings += reduction * self.hourly_rates[hour] * 0.5  # 50% of saved cost as incentive
        
        optimized_cost = np.sum(optimized_profile * self.hourly_rates) - dr_savings
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost) * 100
        
        return {
            'baseline_profile': self.baseline_profile,
            'optimized_profile': optimized_profile,
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'savings': savings,
            'savings_pct': savings_pct,
            'dr_events': dr_events,
            'dr_incentives': dr_savings,
            'method': 'demand_response'
        }

def analyze_cluster_optimization(daily_profiles_df, pricing_models, tou_periods, cluster_column='Cluster'):
    """Analyze optimization opportunities for each cluster"""
    results = {}
    
    for cluster_id in daily_profiles_df[cluster_column].unique():
        cluster_data = daily_profiles_df[daily_profiles_df[cluster_column] == cluster_id]
        avg_profile = cluster_data.iloc[:, :24].mean()
        
        cluster_results = {}
        
        for model_name, model_params in pricing_models.items():
            if model_name == 'flat_rate':
                continue  # Skip flat rate for optimization
            
            optimizer = TOUOptimizer(avg_profile, model_params, tou_periods)
            
            # Run different optimization scenarios
            load_shift_result = optimizer.simulate_load_shifting(shiftable_percentage=0.3)
            battery_result = optimizer.simulate_battery_storage(battery_capacity_kwh=10)
            dr_result = optimizer.simulate_demand_response(reduction_percentage=0.15)
            
            cluster_results[model_name] = {
                'load_shifting': load_shift_result,
                'battery_storage': battery_result,
                'demand_response': dr_result
            }
        
        results[cluster_id] = {
            'cluster_size': len(cluster_data),
            'avg_profile': avg_profile,
            'optimizations': cluster_results
        }
    
    return results

def plot_optimization_results(optimization_results, cluster_id, pricing_model):
    """Plot optimization results for a specific cluster and pricing model"""
    if cluster_id not in optimization_results:
        print(f"Cluster {cluster_id} not found")
        return
    
    if pricing_model not in optimization_results[cluster_id]['optimizations']:
        print(f"Pricing model {pricing_model} not found for cluster {cluster_id}")
        return
    
    data = optimization_results[cluster_id]['optimizations'][pricing_model]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'TOU Optimization Results - Cluster {cluster_id} - {pricing_model}', fontsize=16, fontweight='bold')
    
    hours = range(24)
    
    # 1. Load Shifting
    ax1 = axes[0, 0]
    load_shift = data['load_shifting']
    ax1.plot(hours, load_shift['baseline_profile'], 'b-', label='Baseline', linewidth=2)
    ax1.plot(hours, load_shift['optimized_profile'], 'g--', label='Optimized', linewidth=2)
    ax1.set_title(f'Load Shifting (Savings: {load_shift["savings_pct"]:.1f}%)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Battery Storage
    ax2 = axes[0, 1]
    battery = data['battery_storage']
    ax2.plot(hours, battery['baseline_profile'], 'b-', label='Baseline', linewidth=2)
    ax2.plot(hours, battery['optimized_profile'], 'r--', label='With Battery', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(hours, battery['battery_schedule'], alpha=0.3, color='orange', label='Battery (charge/discharge)')
    ax2.set_title(f'Battery Storage (Savings: {battery["savings_pct"]:.1f}%)')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Power (kW)')
    ax2_twin.set_ylabel('Battery Power (kW)')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Demand Response
    ax3 = axes[1, 0]
    dr = data['demand_response']
    ax3.plot(hours, dr['baseline_profile'], 'b-', label='Baseline', linewidth=2)
    ax3.plot(hours, dr['optimized_profile'], 'm--', label='With DR', linewidth=2)
    # Highlight DR event hours
    for hour in dr['dr_events']:
        ax3.axvspan(hour-0.5, hour+0.5, alpha=0.2, color='red')
    ax3.set_title(f'Demand Response (Savings: {dr["savings_pct"]:.1f}%)')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Power (kW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cost Comparison
    ax4 = axes[1, 1]
    methods = ['Load Shifting', 'Battery Storage', 'Demand Response']
    baseline_costs = [data[method]['baseline_cost'] for method in ['load_shifting', 'battery_storage', 'demand_response']]
    optimized_costs = [data[method]['optimized_cost'] for method in ['load_shifting', 'battery_storage', 'demand_response']]
    savings_pcts = [data[method]['savings_pct'] for method in ['load_shifting', 'battery_storage', 'demand_response']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, baseline_costs, width, label='Baseline Cost', color='lightblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, optimized_costs, width, label='Optimized Cost', color='lightgreen', alpha=0.8)
    
    # Add savings percentage on top of bars
    for i, (bar1, bar2, pct) in enumerate(zip(bars1, bars2, savings_pcts)):
        height = max(bar1.get_height(), bar2.get_height())
        ax4.text(i, height + height*0.02, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Optimization Method')
    ax4.set_ylabel('Daily Cost ($)')
    ax4.set_title('Cost Comparison by Optimization Method')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_optimization_summary(optimization_results):
    """Create a summary DataFrame of all optimization results"""
    summary_data = []
    
    for cluster_id, cluster_data in optimization_results.items():
        for pricing_model, model_data in cluster_data['optimizations'].items():
            for method, result in model_data.items():
                summary_data.append({
                    'Cluster': cluster_id,
                    'Pricing_Model': pricing_model,
                    'Optimization_Method': method,
                    'Baseline_Cost': result['baseline_cost'],
                    'Optimized_Cost': result['optimized_cost'],
                    'Savings_$': result['savings'],
                    'Savings_%': result['savings_pct'],
                    'Cluster_Size': cluster_data['cluster_size']
                })
    
    return pd.DataFrame(summary_data)

def plot_optimization_summary(summary_df):
    """Plot summary of optimization results across all clusters and methods"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TOU Optimization Summary - All Clusters & Methods', fontsize=16, fontweight='bold')
    
    # 1. Savings by cluster and method
    ax1 = axes[0, 0]
    pivot_savings = summary_df.pivot_table(
        index='Cluster', 
        columns='Optimization_Method', 
        values='Savings_%', 
        aggfunc='mean'
    )
    pivot_savings.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Average Savings % by Cluster and Method')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Savings (%)')
    ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Savings by pricing model
    ax2 = axes[0, 1]
    pricing_savings = summary_df.groupby(['Pricing_Model', 'Optimization_Method'])['Savings_%'].mean().unstack()
    pricing_savings.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Average Savings % by Pricing Model')
    ax2.set_xlabel('Pricing Model')
    ax2.set_ylabel('Savings (%)')
    ax2.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Best optimization method per cluster
    ax3 = axes[1, 0]
    best_methods = summary_df.loc[summary_df.groupby(['Cluster', 'Pricing_Model'])['Savings_%'].idxmax()]
    method_counts = best_methods['Optimization_Method'].value_counts()
    method_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
    ax3.set_title('Best Optimization Method Distribution')
    ax3.set_ylabel('')
    
    # 4. Cost savings distribution
    ax4 = axes[1, 1]
    ax4.hist(summary_df['Savings_%'], bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(summary_df['Savings_%'].mean(), color='red', linestyle='--', 
                label=f'Mean: {summary_df["Savings_%"].mean():.1f}%')
    ax4.set_title('Distribution of Savings Percentages')
    ax4.set_xlabel('Savings (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return summary_df

def print_optimization_recommendations(optimization_results):
    """Print recommendations for each cluster"""
    print("=== TOU OPTIMIZATION RECOMMENDATIONS ===\n")
    
    for cluster_id, cluster_data in optimization_results.items():
        print(f"🔹 CLUSTER {cluster_id} ({cluster_data['cluster_size']} days)")
        print(f"   Average daily consumption: {cluster_data['avg_profile'].sum():.2f} kWh")
        print(f"   Peak demand: {cluster_data['avg_profile'].max():.2f} kW\n")
        
        best_options = []
        
        for pricing_model, model_data in cluster_data['optimizations'].items():
            print(f"   📊 {pricing_model.upper()}:")
            
            method_savings = []
            for method, result in model_data.items():
                savings_pct = result['savings_pct']
                savings_dollar = result['savings']
                method_savings.append((method, savings_pct, savings_dollar))
                print(f"      • {method.replace('_', ' ').title()}: {savings_pct:.1f}% (${savings_dollar:.2f}/day)")
            
            best_method = max(method_savings, key=lambda x: x[1])
            best_options.append((pricing_model, best_method[0], best_method[1], best_method[2]))
            print(f"      ⭐ Best: {best_method[0].replace('_', ' ').title()}\n")
        
        # Overall recommendation
        overall_best = max(best_options, key=lambda x: x[2])
        print(f"   🏆 RECOMMENDATION: {overall_best[0]} with {overall_best[1].replace('_', ' ')} ")
        print(f"       💰 Potential savings: {overall_best[2]:.1f}% (${overall_best[3]:.2f}/day)")
        print(f"       📅 Annual savings: ${overall_best[3] * 365:.0f}")
        print("\n" + "="*80 + "\n")