import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Let's simulate the cost for different pricing models using sample hourly load profile data over a day.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ElectricityPricingSimulator:
    """
    A comprehensive electricity pricing simulator for different rate structures
    """
    
    def __init__(self, pricing_models=None, tou_periods=None):
        """
        Initialize simulator with custom pricing models and TOU periods
        
        Parameters:
        pricing_models: dict - Custom pricing model definitions
        tou_periods: dict - Custom TOU period definitions
        """
        # Set default pricing models if not provided
        self.pricing_models = pricing_models or self._get_default_pricing_models()
        
        # Set default TOU periods if not provided
        self.tou_periods = tou_periods or self._get_default_tou_periods()
    
    def _get_default_pricing_models(self):
        """Default pricing model configurations"""
        return {
            'flat_rate': {
                'rate': 0.15,
                'description': 'Fixed rate for all hours'
            },
            'tou_basic': {
                'peak_rate': 0.25,
                'off_peak_rate': 0.10,
                'description': 'Simple peak/off-peak pricing'
            },
            'tou_advanced': {
                'peak_rate': 0.28,
                'shoulder_rate': 0.18,
                'off_peak_rate': 0.08,
                'description': 'Three-tier TOU pricing'
            },
            'demand_charge': {
                'energy_rate': 0.12,
                'demand_rate': 15.0,
                'description': 'Energy + demand charge pricing'
            },
            'tiered': {
                'tier_thresholds': [100, 300, 500],
                'tier_rates': [0.12, 0.15, 0.18, 0.22],
                'description': 'Tiered pricing based on consumption'
            }
        }
    
    def _get_default_tou_periods(self):
        """Default TOU period definitions"""
        return {
            'peak_hours': list(range(16, 22)),  # 4 PM to 9 PM
            'shoulder_hours': list(range(7, 16)) + list(range(22, 24)),  # 7AM-4PM, 10PM-midnight
            'off_peak_hours': list(range(0, 7))  # Midnight to 7 AM
        }
    
    def update_pricing_models(self, new_models):
        """Update pricing models"""
        self.pricing_models.update(new_models)
    
    def update_tou_periods(self, new_periods):
        """Update TOU periods"""
        self.tou_periods.update(new_periods)
    
    def get_tou_rate_for_hour(self, hour, model_name='tou_advanced'):
        """Get the TOU rate for a specific hour and model"""
        model = self.pricing_models[model_name]
        
        if hour in self.tou_periods['peak_hours']:
            return model['peak_rate']
        elif hour in self.tou_periods.get('shoulder_hours', []):
            return model.get('shoulder_rate', model.get('off_peak_rate', 0.10))
        else:
            return model['off_peak_rate']
    
    def calculate_flat_rate_cost(self, load_profile, model_name='flat_rate'):
        """Calculate cost using flat rate pricing"""
        if isinstance(load_profile, pd.Series):
            load_profile = load_profile.values
        
        model = self.pricing_models[model_name]
        rate = model['rate']
        
        total_energy = np.sum(load_profile)
        total_cost = total_energy * rate
        
        return {
            'total_cost': total_cost,
            'energy_cost': total_cost,
            'demand_cost': 0,
            'total_energy': total_energy,
            'pricing_model': f"Flat Rate ({rate:.3f}$/kWh)",
            'model_params': model
        }
    
    def calculate_tou_cost(self, load_profile, model_name='tou_advanced'):
        """Calculate cost using Time-of-Use pricing"""
        if isinstance(load_profile, pd.Series):
            load_profile = load_profile.values
        
        if len(load_profile) != 24:
            raise ValueError("Load profile must have 24 hourly values")
        
        model = self.pricing_models[model_name]
        
        hourly_costs = []
        hourly_rates = []
        
        for hour, load in enumerate(load_profile):
            rate = self.get_tou_rate_for_hour(hour, model_name)
            hour_cost = load * rate
            hourly_costs.append(hour_cost)
            hourly_rates.append(rate)
        
        total_cost = sum(hourly_costs)
        
        return {
            'total_cost': total_cost,
            'energy_cost': total_cost,
            'demand_cost': 0,
            'hourly_costs': hourly_costs,
            'hourly_rates': hourly_rates,
            'pricing_model': f"TOU ({model_name})",
            'model_params': model,
            'tou_periods': self.tou_periods
        }
    
    def calculate_demand_charge_cost(self, load_profile, model_name='demand_charge'):
        """Calculate cost using demand charge pricing"""
        if isinstance(load_profile, pd.Series):
            load_profile = load_profile.values
        
        model = self.pricing_models[model_name]
        
        # Energy cost
        total_energy = np.sum(load_profile)
        energy_cost = total_energy * model['energy_rate']
        
        # Demand cost (based on peak demand)
        peak_demand = np.max(load_profile)
        demand_cost = peak_demand * model['demand_rate']
        
        total_cost = energy_cost + demand_cost
        
        return {
            'total_cost': total_cost,
            'energy_cost': energy_cost,
            'demand_cost': demand_cost,
            'peak_demand': peak_demand,
            'pricing_model': f"Demand Charge",
            'model_params': model
        }
    
    def calculate_tiered_cost(self, load_profile, model_name='tiered'):
        """Calculate cost using tiered pricing"""
        if isinstance(load_profile, pd.Series):
            load_profile = load_profile.values
        
        model = self.pricing_models[model_name]
        tier_thresholds = model['tier_thresholds']
        tier_rates = model['tier_rates']
        
        total_energy = np.sum(load_profile)
        
        # Calculate tiered cost
        remaining_energy = total_energy
        total_cost = 0
        tier_breakdown = []
        
        prev_threshold = 0
        for i, (threshold, rate) in enumerate(zip(tier_thresholds + [float('inf')], tier_rates)):
            if remaining_energy <= 0:
                break
            
            if i == 0:
                energy_in_tier = min(remaining_energy, threshold)
            else:
                energy_in_tier = min(remaining_energy, threshold - prev_threshold)
            
            tier_cost = energy_in_tier * rate
            total_cost += tier_cost
            
            tier_breakdown.append({
                'tier': i + 1,
                'threshold': threshold if threshold != float('inf') else 'Unlimited',
                'rate': rate,
                'energy': energy_in_tier,
                'cost': tier_cost
            })
            
            remaining_energy -= energy_in_tier
            prev_threshold = threshold
        
        return {
            'total_cost': total_cost,
            'energy_cost': total_cost,
            'demand_cost': 0,
            'total_energy': total_energy,
            'tier_breakdown': tier_breakdown,
            'pricing_model': "Tiered Pricing",
            'model_params': model
        }
    
    def compare_pricing_models(self, load_profile, models_to_compare=None):
        """
        Compare specified pricing models for a given load profile
        
        Parameters:
        load_profile: array-like of 24 hourly values
        models_to_compare: list of model names to compare (default: all available)
        """
        if models_to_compare is None:
            models_to_compare = list(self.pricing_models.keys())
        
        results = {}
        comparison_data = []
        
        for model_name in models_to_compare:
            if model_name == 'flat_rate':
                result = self.calculate_flat_rate_cost(load_profile, model_name)
            elif model_name.startswith('tou'):
                result = self.calculate_tou_cost(load_profile, model_name)
            elif model_name == 'demand_charge':
                result = self.calculate_demand_charge_cost(load_profile, model_name)
            elif model_name == 'tiered':
                result = self.calculate_tiered_cost(load_profile, model_name)
            else:
                print(f"Warning: Unknown model type '{model_name}', skipping...")
                continue
            
            results[model_name] = result
            
            comparison_data.append({
                'Model': result['pricing_model'],
                'Total Cost ($)': result['total_cost'],
                'Energy Cost ($)': result['energy_cost'],
                'Demand Cost ($)': result.get('demand_cost', 0)
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate savings vs first model (typically flat rate)
        if len(comparison_df) > 0:
            base_cost = comparison_df.iloc[0]['Total Cost ($)']
            comparison_df['Savings vs Base ($)'] = base_cost - comparison_df['Total Cost ($)']
            comparison_df['Savings vs Base (%)'] = (comparison_df['Savings vs Base ($)'] / base_cost) * 100
        
        return comparison_df, results
    
    def plot_cost_comparison(self, comparison_df, title="Pricing Model Comparison", save_path=None):
        """Plot cost comparison between pricing models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Total costs
        models = comparison_df['Model']
        costs = comparison_df['Total Cost ($)']
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars1 = ax1.bar(range(len(models)), costs, color=colors)
        ax1.set_title(f'{title} - Total Costs')
        ax1.set_ylabel('Total Cost ($)')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            ax1.annotate(f'${cost:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", 
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Savings vs base
        if 'Savings vs Base ($)' in comparison_df.columns:
            savings = comparison_df['Savings vs Base ($)']
            savings_colors = ['gray' if s == 0 else 'green' if s > 0 else 'red' for s in savings]
            
            bars2 = ax2.bar(range(len(models)), savings, color=savings_colors)
            ax2.set_title(f'{title} - Savings vs Base')
            ax2.set_ylabel('Savings ($)')
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for bar, saving in zip(bars2, savings):
                height = bar.get_height()
                ax2.annotate(f'${saving:.2f}', 
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15), 
                            textcoords="offset points", 
                            ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_tou_breakdown(self, load_profile, tou_result, title="TOU Analysis", save_path=None):
        """Plot TOU rate structure and hourly costs"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        hours = range(24)
        
        # Plot 1: Load profile with TOU periods
        ax1.plot(hours, load_profile, 'b-', linewidth=2, marker='o')
        ax1.set_title(f'{title} - Hourly Load Profile')
        ax1.set_ylabel('Load (kW)')
        ax1.grid(True, alpha=0.3)
        
        # Add TOU period backgrounds
        tou_periods = tou_result.get('tou_periods', self.tou_periods)
        ax1.axvspan(min(tou_periods['peak_hours']), max(tou_periods['peak_hours'])+1, 
                   alpha=0.2, color='red', label='Peak Hours')
        if 'shoulder_hours' in tou_periods and tou_periods['shoulder_hours']:
            for start, end in self._get_continuous_periods(tou_periods['shoulder_hours']):
                ax1.axvspan(start, end+1, alpha=0.2, color='orange', label='Shoulder Hours' if start == min(tou_periods['shoulder_hours']) else "")
        for start, end in self._get_continuous_periods(tou_periods['off_peak_hours']):
            ax1.axvspan(start, end+1, alpha=0.2, color='blue', label='Off-Peak Hours' if start == min(tou_periods['off_peak_hours']) else "")
        ax1.legend()
        
        # Plot 2: TOU rates
        rates = tou_result['hourly_rates']
        rate_colors = []
        for h in hours:
            if h in tou_periods['peak_hours']:
                rate_colors.append('red')
            elif h in tou_periods.get('shoulder_hours', []):
                rate_colors.append('orange')
            else:
                rate_colors.append('blue')
        
        ax2.bar(hours, rates, color=rate_colors, alpha=0.7)
        ax2.set_title('TOU Rate Structure')
        ax2.set_ylabel('Rate ($/kWh)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hourly costs
        hourly_costs = tou_result['hourly_costs']
        ax3.bar(hours, hourly_costs, color=rate_colors, alpha=0.7)
        ax3.set_title('Hourly Electricity Costs')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Cost ($)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _get_continuous_periods(self, hours_list):
        """Helper function to get continuous periods from hour list"""
        if not hours_list:
            return []
        
        hours_list = sorted(hours_list)
        periods = []
        start = hours_list[0]
        prev = start
        
        for hour in hours_list[1:] + [hours_list[-1] + 2]:  # Add sentinel
            if hour != prev + 1:
                periods.append((start, prev))
                start = hour
            prev = hour
        
        return periods

# Convenience functions
def create_pricing_simulator(pricing_models=None, tou_periods=None):
    """Create a pricing simulator with custom models and periods"""
    return ElectricityPricingSimulator(pricing_models, tou_periods)

def analyze_cluster_pricing(daily_profiles_df, cluster_column='Cluster', 
                          pricing_models=None, tou_periods=None, models_to_compare=None):
    """
    Analyze pricing for each cluster with custom pricing models
    
    Parameters:
    daily_profiles_df: DataFrame with daily profiles and cluster labels
    cluster_column: name of cluster column
    pricing_models: dict with custom pricing models
    tou_periods: dict with custom TOU periods
    models_to_compare: list of model names to compare
    """
    simulator = ElectricityPricingSimulator(pricing_models, tou_periods)
    cluster_results = {}
    
    for cluster_id in daily_profiles_df[cluster_column].unique():
        print(f"Analyzing Cluster {cluster_id}...")
        
        # Get average profile for cluster
        cluster_data = daily_profiles_df[daily_profiles_df[cluster_column] == cluster_id]
        avg_profile = cluster_data.iloc[:, :24].mean()  # First 24 columns are hours
        
        # Analyze pricing
        comparison_df, detailed_results = simulator.compare_pricing_models(avg_profile, models_to_compare)
        
        cluster_results[cluster_id] = {
            'cluster_size': len(cluster_data),
            'avg_profile': avg_profile,
            'comparison_df': comparison_df,
            'detailed_results': detailed_results,
            'simulator': simulator
        }
    
    return cluster_results

def plot_cluster_pricing_analysis(cluster_results):
    """Plot pricing analysis for all clusters"""
    n_clusters = len(cluster_results)
    fig, axes = plt.subplots(n_clusters, 2, figsize=(15, 5*n_clusters))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for i, (cluster_id, results) in enumerate(cluster_results.items()):
        # Plot 1: Load profile
        hours = range(24)
        axes[i, 0].plot(hours, results['avg_profile'], 'b-', linewidth=2, marker='o')
        axes[i, 0].set_title(f'Cluster {cluster_id} Average Load Profile\n({results["cluster_size"]} days)')
        axes[i, 0].set_xlabel('Hour of Day')
        axes[i, 0].set_ylabel('Average Load (kW)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add TOU period backgrounds if available
        if 'simulator' in results:
            tou_periods = results['simulator'].tou_periods
            axes[i, 0].axvspan(min(tou_periods['peak_hours']), max(tou_periods['peak_hours'])+1, 
                              alpha=0.2, color='red', label='Peak')
            if 'shoulder_hours' in tou_periods and tou_periods['shoulder_hours']:
                axes[i, 0].axvspan(min(tou_periods['shoulder_hours']), max(tou_periods['shoulder_hours'])+1, 
                                  alpha=0.2, color='orange', label='Shoulder')
            axes[i, 0].axvspan(min(tou_periods['off_peak_hours']), max(tou_periods['off_peak_hours'])+1, 
                              alpha=0.2, color='blue', label='Off-Peak')
            axes[i, 0].legend(fontsize=8)
        
        # Plot 2: Cost comparison
        comparison_df = results['comparison_df']
        models = comparison_df['Model']
        costs = comparison_df['Total Cost ($)']
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = axes[i, 1].bar(range(len(models)), costs, color=colors)
        axes[i, 1].set_title(f'Cluster {cluster_id} Pricing Comparison')
        axes[i, 1].set_ylabel('Daily Cost ($)')
        axes[i, 1].set_xticks(range(len(models)))
        axes[i, 1].set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            axes[i, 1].annotate(f'${cost:.2f}', 
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points", 
                              ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def calculate_spain_pvpc_cost(load_profile, eu_pricing_models, eu_tou_periods):
    """Calculate cost using Spain's PVPC three-tier system"""
    model = eu_pricing_models['spain_valley']
    periods = eu_tou_periods['spain_pvpc']
    
    total_cost = 0
    hourly_costs = []
    
    for hour, load in enumerate(load_profile):
        if hour in periods['valley_hours']:
            rate = model['valley_rate']
        elif hour in periods['flat_hours']:
            rate = model['flat_rate']
        else:  # peak hours
            rate = model['peak_rate']
        
        hour_cost = load * rate
        total_cost += hour_cost
        hourly_costs.append(hour_cost)
    
    return {
        'total_cost': total_cost,
        'hourly_costs': hourly_costs,
        'model': 'Spain PVPC'
    }

def analyze_with_eu_pricing(daily_profiles_df, eu_pricing_models, eu_tou_periods, cluster_column='Cluster'):
    """Analyze clusters with EU pricing - comparing flat rate vs TOU within each country"""
    results = {}
    
    # Define flat rate baselines for each country
    country_flat_rates = {
        'Germany': 0.32,  # Germany uses mostly flat rate
        'France': 0.22,   # Typical French flat rate baseline
        'Netherlands': 0.23,  # Typical Dutch flat rate baseline  
        'Spain': 0.18     # Typical Spanish flat rate baseline
    }
    
    for cluster_id in daily_profiles_df[cluster_column].unique():
        cluster_data = daily_profiles_df[daily_profiles_df[cluster_column] == cluster_id]
        avg_profile = cluster_data.iloc[:, :24].mean()  # Average hourly profile
        
        # Germany - Only flat rate (no widespread TOU)
        germany_flat = avg_profile.sum() * country_flat_rates['Germany']
        
        # France - Compare flat rate vs Tempo (Blue/Red days)
        france_flat = avg_profile.sum() * country_flat_rates['France']
        france_blue_tou = sum([
            avg_profile[h] * eu_pricing_models['france_blue_day']['peak_rate'] 
            if 6 <= h <= 21 else avg_profile[h] * eu_pricing_models['france_blue_day']['off_peak_rate']
            for h in range(24)
        ])
        france_red_tou = sum([
            avg_profile[h] * eu_pricing_models['france_red_day']['peak_rate'] 
            if 6 <= h <= 21 else avg_profile[h] * eu_pricing_models['france_red_day']['off_peak_rate']
            for h in range(24)
        ])
        
        # Netherlands - Compare flat rate vs Capacity pricing
        netherlands_flat = avg_profile.sum() * country_flat_rates['Netherlands']
        netherlands_energy = avg_profile.sum() * eu_pricing_models['netherlands_capacity']['energy_rate']
        netherlands_demand = avg_profile.max() * eu_pricing_models['netherlands_capacity']['demand_rate']
        netherlands_capacity = netherlands_energy + netherlands_demand
        
        # Spain - Compare flat rate vs PVPC
        spain_flat = avg_profile.sum() * country_flat_rates['Spain']
        spain_result = calculate_spain_pvpc_cost(avg_profile, eu_pricing_models, eu_tou_periods)
        spain_pvpc = spain_result['total_cost']
        
        results[cluster_id] = {
            'cluster_size': len(cluster_data),
            'daily_consumption_kwh': avg_profile.sum(),
            'peak_demand_kw': avg_profile.max(),
            'avg_profile': avg_profile,
            'countries': {
                'Germany': {
                    'flat_rate': germany_flat,
                    'tou_available': False,
                    'note': 'Germany primarily uses flat rate pricing'
                },
                'France': {
                    'flat_rate': france_flat,
                    'blue_day_tempo': france_blue_tou,
                    'red_day_tempo': france_red_tou,
                    'blue_savings_eur': france_flat - france_blue_tou,
                    'blue_savings_pct': ((france_flat - france_blue_tou) / france_flat * 100),
                    'red_penalty_eur': france_red_tou - france_flat,
                    'red_penalty_pct': ((france_red_tou - france_flat) / france_flat * 100)
                },
                'Netherlands': {
                    'flat_rate': netherlands_flat,
                    'capacity_pricing': netherlands_capacity,
                    'savings_eur': netherlands_flat - netherlands_capacity,
                    'savings_pct': ((netherlands_flat - netherlands_capacity) / netherlands_flat * 100)
                },
                'Spain': {
                    'flat_rate': spain_flat,
                    'pvpc_tou': spain_pvpc,
                    'savings_eur': spain_flat - spain_pvpc,
                    'savings_pct': ((spain_flat - spain_pvpc) / spain_flat * 100)
                }
            }
        }
    
    return results

def plot_eu_pricing_analysis(eu_results, eu_tou_periods):
    """Plot EU pricing analysis - comparing flat rate vs TOU within each country"""
    n_clusters = len(eu_results)
    clusters = list(eu_results.keys())
    
    # Create subplots - one for each country
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EU Country Pricing Comparison: Flat Rate vs TOU Models by Cluster', fontsize=16, fontweight='bold')
    
    # 1. France: Flat vs Tempo (Blue/Red days)
    ax1 = axes[0, 0]
    france_flat = [eu_results[c]['countries']['France']['flat_rate'] for c in clusters]
    france_blue = [eu_results[c]['countries']['France']['blue_day_tempo'] for c in clusters]
    france_red = [eu_results[c]['countries']['France']['red_day_tempo'] for c in clusters]
    
    x = np.arange(len(clusters))
    width = 0.25
    
    ax1.bar(x - width, france_flat, width, label='Flat Rate (€0.22/kWh)', color='#1f77b4', alpha=0.8)
    ax1.bar(x, france_blue, width, label='Tempo Blue Day', color='#2ca02c', alpha=0.8)
    ax1.bar(x + width, france_red, width, label='Tempo Red Day', color='#d62728', alpha=0.8)
    
    # Add savings/penalty percentages on top
    for i, c in enumerate(clusters):
        blue_pct = eu_results[c]['countries']['France']['blue_savings_pct']
        red_pct = eu_results[c]['countries']['France']['red_penalty_pct']
        ax1.text(i, france_blue[i] + 0.2, f'{blue_pct:+.1f}%', ha='center', va='bottom', fontsize=8, color='green' if blue_pct > 0 else 'red')
        ax1.text(i + width, france_red[i] + 0.2, f'{red_pct:+.1f}%', ha='center', va='bottom', fontsize=8, color='red')
    
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Daily Cost (€)')
    ax1.set_title('🇫🇷 France: Flat Rate vs EDF Tempo')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{c}' for c in clusters])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Netherlands: Flat vs Capacity
    ax2 = axes[0, 1]
    nl_flat = [eu_results[c]['countries']['Netherlands']['flat_rate'] for c in clusters]
    nl_capacity = [eu_results[c]['countries']['Netherlands']['capacity_pricing'] for c in clusters]
    
    x = np.arange(len(clusters))
    width = 0.35
    
    ax2.bar(x - width/2, nl_flat, width, label='Flat Rate (€0.23/kWh)', color='#ff7f0e', alpha=0.8)
    ax2.bar(x + width/2, nl_capacity, width, label='Capacity Pricing', color='#9467bd', alpha=0.8)
    
    # Add savings percentages
    for i, c in enumerate(clusters):
        savings_pct = eu_results[c]['countries']['Netherlands']['savings_pct']
        ax2.text(i + width/2, nl_capacity[i] + 0.2, f'{savings_pct:+.1f}%', ha='center', va='bottom', fontsize=8, color='green' if savings_pct > 0 else 'red')
    
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Daily Cost (€)')
    ax2.set_title('🇳🇱 Netherlands: Flat Rate vs Capacity Pricing')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{c}' for c in clusters])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Spain: Flat vs PVPC
    ax3 = axes[1, 0]
    spain_flat = [eu_results[c]['countries']['Spain']['flat_rate'] for c in clusters]
    spain_pvpc = [eu_results[c]['countries']['Spain']['pvpc_tou'] for c in clusters]
    
    x = np.arange(len(clusters))
    width = 0.35
    
    ax3.bar(x - width/2, spain_flat, width, label='Flat Rate (€0.18/kWh)', color='#8c564b', alpha=0.8)
    ax3.bar(x + width/2, spain_pvpc, width, label='PVPC TOU', color='#e377c2', alpha=0.8)
    
    # Add savings percentages
    for i, c in enumerate(clusters):
        savings_pct = eu_results[c]['countries']['Spain']['savings_pct']
        ax3.text(i + width/2, spain_pvpc[i] + 0.2, f'{savings_pct:+.1f}%', ha='center', va='bottom', fontsize=8, color='green' if savings_pct > 0 else 'red')
    
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Daily Cost (€)')
    ax3.set_title('🇪🇸 Spain: Flat Rate vs PVPC TOU')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C{c}' for c in clusters])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary: Best savings opportunity by country and cluster
    ax4 = axes[1, 1]
    
    # Get best savings for each country/cluster
    countries = ['France\n(Blue Day)', 'Netherlands', 'Spain']
    country_keys = ['France', 'Netherlands', 'Spain']
    savings_keys = ['blue_savings_pct', 'savings_pct', 'savings_pct']
    
    cluster_savings = {}
    for i, c in enumerate(clusters):
        cluster_savings[f'C{c}'] = [
            eu_results[c]['countries']['France']['blue_savings_pct'],
            eu_results[c]['countries']['Netherlands']['savings_pct'],
            eu_results[c]['countries']['Spain']['savings_pct']
        ]
    
    x = np.arange(len(countries))
    width = 0.8 / len(clusters)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    for i, (cluster_name, savings) in enumerate(cluster_savings.items()):
        ax4.bar(x + i * width, savings, width, label=cluster_name, color=colors[i], alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Country')
    ax4.set_ylabel('Savings vs Flat Rate (%)')
    ax4.set_title('💰 TOU Savings Potential by Country & Cluster')
    ax4.set_xticks(x + width * (len(clusters) - 1) / 2)
    ax4.set_xticklabels(countries)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_eu_pricing_results(eu_results):
    """Print detailed EU pricing results with country-specific comparisons"""
    print("=== EU COUNTRY-SPECIFIC PRICING ANALYSIS ===")
    print("Comparing flat rate vs TOU models within each country\n")
    
    for cluster_id, data in eu_results.items():
        print(f"🔹 CLUSTER {cluster_id} ({data['cluster_size']} days)")
        print(f"   Daily consumption: {data['daily_consumption_kwh']:.2f} kWh")
        print(f"   Peak demand: {data['peak_demand_kw']:.2f} kW\n")
        
        # Germany
        germany = data['countries']['Germany']
        print(f"🇩🇪 GERMANY:")
        print(f"   • Flat rate only: €{germany['flat_rate']:.2f}/day")
        print(f"   • {germany['note']}\n")
        
        # France
        france = data['countries']['France']
        print(f"🇫🇷 FRANCE (EDF Tempo):")
        print(f"   • Flat rate baseline: €{france['flat_rate']:.2f}/day")
        print(f"   • Blue day Tempo: €{france['blue_day_tempo']:.2f}/day (savings: {france['blue_savings_pct']:+.1f}%)")
        print(f"   • Red day Tempo: €{france['red_day_tempo']:.2f}/day (penalty: {france['red_penalty_pct']:+.1f}%)")
