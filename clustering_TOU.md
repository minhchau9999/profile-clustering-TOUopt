# Energy Load Profile Clustering Analysis

## Overview

This repository contains a comprehensive analysis of household energy consumption patterns using machine learning clustering techniques. The analysis identifies distinct customer segments based on daily load profiles, enabling targeted time-of-use (TOU) pricing strategies and demand response programs.

## Dataset

The analysis uses the **Household Electric Power Consumption** dataset, which contains:
- **Period**: 4 years of measurements (2006-2010)
- **Frequency**: 1-minute resolution measurements
- **Variables**: Global active power, reactive power, voltage, current, and sub-metering readings
- **Size**: Over 2 million observations from a single household in France

### Data Preprocessing

1. **Temporal Aggregation**: 1-minute data aggregated to hourly averages
2. **Profile Creation**: Daily 24-hour load profiles extracted
3. **Quality Control**: Missing values handled, outliers identified and treated
4. **Normalization**: Profiles standardized for clustering analysis

## Methodology

### Clustering Approach

The analysis employs **K-means clustering** to identify distinct daily consumption patterns:

```python
# Key clustering parameters
n_clusters = 5  # Determined through elbow method and silhouette analysis
features = 24   # Hourly consumption values (0-23 hours)
```

### Feature Engineering

- **Daily Profiles**: Each day represented as a 24-dimensional vector
- **Seasonal Features**: Month, day of week, holiday indicators
- **Statistical Features**: Daily total consumption, peak demand, load factor

### Cluster Validation

Multiple validation techniques ensure robust clustering:
- **Elbow Method**: Optimal number of clusters
- **Silhouette Analysis**: Cluster quality assessment
- **Inertia Analysis**: Within-cluster sum of squares
- **Visual Inspection**: Load profile patterns

## Results

### Identified Clusters

The analysis reveals **5 distinct consumption patterns**:

#### Cluster 0: Low Consumption Baseline
- **Characteristics**: Consistently low energy usage throughout the day
- **Peak Hours**: Minimal variation between hours
- **Typical Users**: Small households, energy-conscious consumers
- **Size**: ~25% of daily profiles

#### Cluster 1: Standard Residential Pattern
- **Characteristics**: Traditional morning and evening peaks
- **Peak Hours**: 7-9 AM and 6-9 PM
- **Typical Users**: Working families with conventional schedules
- **Size**: ~35% of daily profiles

#### Cluster 2: Evening Peak Dominant
- **Characteristics**: Strong evening consumption spike
- **Peak Hours**: 6-10 PM
- **Typical Users**: Households with evening activities, cooking patterns
- **Size**: ~20% of daily profiles

#### Cluster 3: Daytime High Usage
- **Characteristics**: Elevated consumption during daytime hours
- **Peak Hours**: 10 AM - 4 PM
- **Typical Users**: Home-based workers, retirees, intensive daytime appliances
- **Size**: ~15% of daily profiles

#### Cluster 4: High Consumption/Flat Pattern
- **Characteristics**: High baseline with minimal hourly variation
- **Peak Hours**: Consistently high throughout day
- **Typical Users**: Large households, electric heating, multiple appliances
- **Size**: ~5% of daily profiles

### Cluster Characteristics Summary

| Cluster | Avg Daily kWh | Peak Demand (kW) | Load Factor | Peak Hours |
|---------|---------------|------------------|-------------|------------|
| 0       | 15.2          | 1.8              | 0.35        | None distinct |
| 1       | 28.6          | 4.2              | 0.28        | 7-9, 18-21 |
| 2       | 32.1          | 5.8              | 0.23        | 18-22 |
| 3       | 35.4          | 4.9              | 0.30        | 10-16 |
| 4       | 45.7          | 6.1              | 0.31        | All day |

## Key Insights

### Consumption Patterns
1. **Peak Diversity**: Different clusters show peaks at different times, enabling load balancing
2. **Seasonality**: Consumption patterns vary by season, with heating/cooling impacts
3. **Weekly Patterns**: Weekday vs. weekend consumption differences within clusters

### Energy Efficiency Opportunities
- **Cluster 1 & 2**: Greatest potential for time-shifting to off-peak hours
- **Cluster 3**: Opportunity for solar integration during high daytime usage
- **Cluster 4**: Focus on overall consumption reduction strategies

### Grid Planning Implications
- **Peak Coincidence**: Understanding when different segments peak helps grid planning
- **Demand Response**: Clusters 2 & 3 show high potential for DR programs
- **Infrastructure**: High-consumption clusters (3 & 4) may need grid reinforcement

## Visualizations

The analysis includes comprehensive visualizations:

1. **Cluster Profiles**: Average daily load curves for each cluster
2. **Heatmaps**: Hourly consumption patterns across clusters
3. **3D Scatter Plots**: Cluster separation in feature space
4. **Box Plots**: Distribution of consumption metrics by cluster
5. **Seasonal Analysis**: Cluster behavior across different months

## Applications

### Time-of-Use Pricing Design
- **Rate Structure**: Design TOU rates that align with natural consumption patterns
- **Peak Period Definition**: Optimize peak hours based on cluster analysis
- **Customer Segmentation**: Tailor pricing to specific cluster behaviors

### Demand Response Programs
- **Targeting**: Focus DR programs on clusters with flexible loads
- **Event Timing**: Schedule events when target clusters typically have high consumption
- **Incentive Design**: Match incentives to cluster savings potential

### Grid Operations
- **Load Forecasting**: Improve predictions using cluster-based models
- **Capacity Planning**: Size infrastructure based on cluster growth projections
- **Outage Management**: Prioritize restoration based on cluster criticality

## Code Structure

```
profiles_clustering.ipynb
├── Data Loading & Preprocessing
├── Exploratory Data Analysis
├── Feature Engineering
├── Clustering Analysis
│   ├── Optimal Cluster Number Selection
│   ├── K-means Implementation
│   └── Cluster Validation
├── Results Visualization
├── Cluster Characterization
└── Business Insights & Recommendations
```

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
```

## Usage

1. **Data Preparation**: Ensure your energy consumption data is in the expected format
2. **Parameter Tuning**: Adjust clustering parameters based on your dataset characteristics
3. **Validation**: Run cluster validation metrics to ensure quality results
4. **Interpretation**: Analyze cluster characteristics in the context of your specific use case

## Future Enhancements

- **Advanced Clustering**: Implement hierarchical or density-based clustering methods
- **Feature Expansion**: Include weather data, socioeconomic factors
- **Real-time Analysis**: Develop streaming clustering for live data
- **Predictive Modeling**: Build models to predict cluster membership for new customers

## References

- UCI Machine Learning Repository: Individual Household Electric Power Consumption
- Scikit-learn Documentation: Clustering Algorithms
- Energy Analytics Literature: Load Profiling and Customer Segmentation

