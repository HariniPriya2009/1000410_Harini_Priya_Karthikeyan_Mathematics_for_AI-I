"""
Smart Elevator Predictive Maintenance Dashboard
A production-ready Streamlit application for elevator system monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Smart Elevator Predictive Maintenance Dashboard",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# GLOBAL VARIABLES
# ============================================
# Define numeric columns globally to avoid NameError
numeric_columns = ['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0a1628 0%, #0d2137 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        metric-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(145deg, #1e3a5f 0%, #0f2744 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), 
                    0 0 20px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4), 
                    0 0 30px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.4);
    }
    
    /* Section styling */
    .section-container {
        background: rgba(15, 39, 68, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #00d4ff;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0d2137 0%, #0a1628 100%);
    }
    
    /* Success, warning, info message styling */
    .stSuccess {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stWarning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stInfo {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1628;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e3a5f;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00d4ff;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: rgba(15, 39, 68, 0.3) !important;
        border-radius: 8px;
    }
    
    .stDataFrame table {
        color: #e2e8f0 !important;
        background-color: rgba(15, 39, 68, 0.5) !important;
    }
    
    .stDataFrame th {
        background-color: rgba(0, 212, 255, 0.3) !important;
        color: #ffffff !important;
    }
    
    .stDataFrame td {
        background-color: rgba(15, 39, 68, 0.5) !important;
        color: #e2e8f0 !important;
        border-color: rgba(0, 212, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING & CLEANING
# ============================================
@st.cache_data
def load_and_clean_data(file_path):
    """
    Load and clean the elevator maintenance dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing vibration values using interpolation
    df['vibration'] = df['vibration'].interpolate(method='linear', limit_direction='both')
    
    # Ensure numeric columns are correct type
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any remaining NaN values with mean
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    return df

# ============================================
# DATA LOADING
# ============================================
try:
    df = load_and_clean_data('Elevator predictive-maintenance-dataset.csv')
    st.session_state['data_loaded'] = True
except FileNotFoundError:
    st.error("Dataset file 'Elevator predictive-maintenance-dataset.csv' not found. Please ensure the file is in the same directory as app.py")
    st.session_state['data_loaded'] = False
    st.stop()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.markdown("""
<div style='padding: 1rem; background: rgba(0, 212, 255, 0.1); border-radius: 12px; margin-bottom: 1.5rem;'>
    <h3 style='color: #00d4ff; margin: 0;'>🎛️ Control Panel</h3>
    <p style='color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Adjust filters to analyze specific data ranges</p>
</div>
""", unsafe_allow_html=True)

# Revolutions range slider
min_rev, max_rev = float(df['revolutions'].min()), float(df['revolutions'].max())
rev_range = st.sidebar.slider(
    "🔄 Revolutions Range",
    min_value=min_rev,
    max_value=max_rev,
    value=(min_rev, max_rev),
    step=10.0,
    help="Filter data based on revolutions"
)

# Humidity range slider
min_hum, max_hum = float(df['humidity'].min()), float(df['humidity'].max())
hum_range = st.sidebar.slider(
    "💧 Humidity Range (%)",
    min_value=min_hum,
    max_value=max_hum,
    value=(min_hum, max_hum),
    step=1.0,
    help="Filter data based on humidity levels"
)

# Vibration threshold slider
vib_threshold = st.sidebar.slider(
    "📊 Vibration Threshold",
    min_value=float(df['vibration'].min()),
    max_value=float(df['vibration'].max()),
    value=float(df['vibration'].quantile(0.95)),
    step=0.01,
    help="Set threshold for anomaly detection"
)

# Reset filters button
if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
    st.session_state['reset_filters'] = True
else:
    st.session_state['reset_filters'] = False

# Apply filters
if st.session_state['reset_filters']:
    filtered_df = df.copy()
else:
    filtered_df = df[
        (df['revolutions'] >= rev_range[0]) & 
        (df['revolutions'] <= rev_range[1]) &
        (df['humidity'] >= hum_range[0]) & 
        (df['humidity'] <= hum_range[1])
    ].copy()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class='header-title'>🛗 Smart Elevator Predictive Maintenance</div>
<div class='header-subtitle'>Advanced Analytics Dashboard for Real-Time System Monitoring</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SECTION 1: KPI METRICS
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>📈 Key Performance Indicators</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_vib = filtered_df['vibration'].mean()
    st.metric(
        label="Average Vibration",
        value=f"{avg_vib:.4f}",
        delta=f"{(avg_vib/df['vibration'].mean()*100)-100:.1f}%",
        help="Mean vibration level across all readings"
    )

with col2:
    max_vib = filtered_df['vibration'].max()
    st.metric(
        label="Max Vibration",
        value=f"{max_vib:.4f}",
        delta=f"{(max_vib/df['vibration'].max()*100)-100:.1f}%",
        help="Maximum vibration reading"
    )

with col3:
    avg_rev = filtered_df['revolutions'].mean()
    st.metric(
        label="Avg Revolutions",
        value=f"{avg_rev:.2f}",
        delta=f"{(avg_rev/df['revolutions'].mean()*100)-100:.1f}%",
        help="Average revolutions per reading"
    )

with col4:
    avg_hum = filtered_df['humidity'].mean()
    st.metric(
        label="Avg Humidity",
        value=f"{avg_hum:.2f}%",
        delta=f"{(avg_hum/df['humidity'].mean()*100)-100:.1f}%",
        help="Average humidity percentage"
    )

# ============================================
# SECTION 2: TIME SERIES ANALYSIS
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>⏱️ Time Series Analysis</div>
</div>
""", unsafe_allow_html=True)

# Calculate statistics for anomaly highlighting
vib_mean = filtered_df['vibration'].mean()
vib_std = filtered_df['vibration'].std()
vib_upper = vib_mean + 2 * vib_std
vib_lower = vib_mean - 2 * vib_std

# Create time series plot
fig_ts = go.Figure()

# Add main line
fig_ts.add_trace(go.Scatter(
    x=filtered_df['ID'],
    y=filtered_df['vibration'],
    mode='lines',
    name='Vibration',
    line=dict(color='#00d4ff', width=1),
    hovertemplate='<b>ID: %{x}</b><br>Vibration: %{y:.4f}<extra></extra>'
))

# Highlight abnormal spikes
anomalies = filtered_df[
    (filtered_df['vibration'] > vib_upper) | 
    (filtered_df['vibration'] < vib_lower)
]

if not anomalies.empty:
    fig_ts.add_trace(go.Scatter(
        x=anomalies['ID'],
        y=anomalies['vibration'],
        mode='markers',
        name='Anomalies',
        marker=dict(
            color='#ff4757',
            size=8,
            symbol='diamond',
            line=dict(color='#fff', width=1)
        ),
        hovertemplate='<b>Anomaly Detected</b><br>ID: %{x}<br>Vibration: %{y:.4f}<extra></extra>'
    ))

# Add threshold lines
fig_ts.add_hline(y=vib_upper, line_dash="dash", line_color="#ff4757", annotation_text="Upper Limit")
fig_ts.add_hline(y=vib_lower, line_dash="dash", line_color="#ff4757", annotation_text="Lower Limit")

fig_ts.update_layout(
    title="Vibration Levels Over Time",
    xaxis_title="Reading ID",
    yaxis_title="Vibration Level",
    hovermode='x unified',
    template='plotly_dark',
    height=400,
    plot_bgcolor='rgba(15, 39, 68, 0.3)',
    paper_bgcolor='rgba(15, 39, 68, 0.3)',
    font=dict(color='#e2e8f0'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_ts, use_container_width=True)

# ============================================
# SECTION 3: DISTRIBUTION ANALYSIS
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>📊 Distribution Analysis</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_rev_dist = px.histogram(
        filtered_df,
        x='revolutions',
        nbins=50,
        title='Distribution of Revolutions',
        color_discrete_sequence=['#00d4ff'],
        template='plotly_dark'
    )
    
    fig_rev_dist.update_layout(
        xaxis_title="Revolutions",
        yaxis_title="Count",
        height=350,
        plot_bgcolor='rgba(15, 39, 68, 0.3)',
        paper_bgcolor='rgba(15, 39, 68, 0.3)',
        font=dict(color='#e2e8f0')
    )
    
    fig_rev_dist.update_traces(
        hovertemplate='<b>Revolutions: %{x:.2f}</b><br>Count: %{y}<extra></extra>'
    )
    
    st.plotly_chart(fig_rev_dist, use_container_width=True)

with col2:
    fig_hum_dist = px.histogram(
        filtered_df,
        x='humidity',
        nbins=50,
        title='Distribution of Humidity',
        color_discrete_sequence=['#10b981'],
        template='plotly_dark'
    )
    
    fig_hum_dist.update_layout(
        xaxis_title="Humidity (%)",
        yaxis_title="Count",
        height=350,
        plot_bgcolor='rgba(15, 39, 68, 0.3)',
        paper_bgcolor='rgba(15, 39, 68, 0.3)',
        font=dict(color='#e2e8f0')
    )
    
    fig_hum_dist.update_traces(
        hovertemplate='<b>Humidity: %{x:.2f}%</b><br>Count: %{y}<extra></extra>'
    )
    
    st.plotly_chart(fig_hum_dist, use_container_width=True)

# ============================================
# SECTION 4: RELATIONSHIP ANALYSIS
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🔗 Relationship Analysis</div>
</div>
""", unsafe_allow_html=True)

# Calculate correlation and regression
corr_coef = filtered_df['revolutions'].corr(filtered_df['vibration'])
slope, intercept, r_value, p_value, std_err = linregress(
    filtered_df['revolutions'],
    filtered_df['vibration']
)

# Create scatter plot with regression line
fig_scatter = px.scatter(
    filtered_df,
    x='revolutions',
    y='vibration',
    color='humidity',
    title='Revolutions vs Vibration Relationship',
    color_continuous_scale='Viridis',
    template='plotly_dark',
    hover_data=['ID']
)

# Add OLS regression trendline
x_range = np.array([filtered_df['revolutions'].min(), filtered_df['revolutions'].max()])
y_range = slope * x_range + intercept

fig_scatter.add_trace(go.Scatter(
    x=x_range,
    y=y_range,
    mode='lines',
    name=f'OLS Trendline (r={corr_coef:.3f})',
    line=dict(color='#ff4757', width=3, dash='dash')
))

fig_scatter.update_layout(
    xaxis_title="Revolutions",
    yaxis_title="Vibration",
    height=450,
    plot_bgcolor='rgba(15, 39, 68, 0.3)',
    paper_bgcolor='rgba(15, 39, 68, 0.3)',
    font=dict(color='#e2e8f0'),
    coloraxis_colorbar=dict(title="Humidity (%)")
)

fig_scatter.update_traces(
    hovertemplate='<b>Revolutions: %{x:.2f}</b><br>Vibration: %{y:.4f}<br>Humidity: %{marker.color:.2f}%<extra></extra>'
)

st.plotly_chart(fig_scatter, use_container_width=True)

# Display correlation information
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📊 Pearson Correlation: **{corr_coef:.4f}**")
with col2:
    st.info(f"📈 R-Squared: **{r_value**2:.4f}**")
with col3:
    st.info(f"⚡ P-Value: **{p_value:.4e}**")

# ============================================
# SECTION 5: SENSOR HEALTH CHECK
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🔬 Sensor Health Check</div>
</div>
""", unsafe_allow_html=True)

# Prepare sensor data
sensor_columns = ['x1', 'x2', 'x3', 'x4', 'x5']
sensor_data = filtered_df[sensor_columns].melt(var_name='Sensor', value_name='Reading')

# Create box plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=sensor_data,
    x='Sensor',
    y='Reading',
    palette='viridis',
    ax=ax,
    showfliers=True
)

# Customize plot
ax.set_xlabel('Sensor ID', fontsize=12, color='white')
ax.set_ylabel('Reading Value', fontsize=12, color='white')
ax.set_title('Sensor Reading Distribution with Outliers', fontsize=14, color='#00d4ff', pad=20)

# Style the plot
ax.spines['bottom'].set_color('#4a5568')
ax.spines['top'].set_color('#4a5568')
ax.spines['left'].set_color('#4a5568')
ax.spines['right'].set_color('#4a5568')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('#00d4ff')

# Set background
ax.set_facecolor('#0d2137')
fig.patch.set_facecolor('#000000')

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# Calculate outliers for each sensor
st.markdown("### 📋 Outlier Detection Summary")
outlier_info = []
for sensor in sensor_columns:
    Q1 = filtered_df[sensor].quantile(0.25)
    Q3 = filtered_df[sensor].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = filtered_df[(filtered_df[sensor] < lower_bound) | (filtered_df[sensor] > upper_bound)]
    outlier_info.append({
        'Sensor': sensor.upper(),
        'Outliers': len(outliers),
        'Percentage': f"{(len(outliers)/len(filtered_df)*100):.2f}%"
    })

outlier_df = pd.DataFrame(outlier_info)
st.dataframe(
    outlier_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Sensor': st.column_config.TextColumn('Sensor ID', width='medium'),
        'Outliers': st.column_config.NumberColumn('Count', width='small'),
        'Percentage': st.column_config.TextColumn('Percentage', width='small')
    }
)

# ============================================
# SECTION 6: CORRELATION HEATMAP
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🌡️ Correlation Heatmap</div>
</div>
""", unsafe_allow_html=True)

# Calculate correlation matrix
corr_matrix = filtered_df[numeric_columns].corr()

# Create heatmap
fig_heatmap, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={'shrink': 0.8},
    ax=ax,
    annot_kws={'size': 10, 'color': 'black', 'weight': 'bold'}
)

# Customize plot
ax.set_title('Feature Correlation Matrix', fontsize=16, color='#00d4ff', pad=20)
ax.tick_params(axis='x', colors='white', rotation=45, ha='right')
ax.tick_params(axis='y', colors='white', rotation=0)
fig_heatmap.patch.set_facecolor('#000000')

plt.tight_layout()
st.pyplot(fig_heatmap, use_container_width=True)

# ============================================
# SECTION 7: ANOMALY DETECTION
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🚨 Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# Detect anomalies based on threshold
anomalies_df = filtered_df[filtered_df['vibration'] > vib_threshold].copy()
anomaly_count = len(anomalies_df)
anomaly_percentage = (anomaly_count / len(filtered_df)) * 100

# Display anomaly statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.error(f"🚨 Anomalies Detected: **{anomaly_count}**")
with col2:
    st.warning(f"⚠️ Percentage: **{anomaly_percentage:.2f}%**")
with col3:
    st.info(f"📊 Threshold: **{vib_threshold:.4f}**")

# Create anomaly visualization
if anomaly_count > 0:
    fig_anomaly = go.Figure()
    
    # Add normal readings
    fig_anomaly.add_trace(go.Scatter(
        x=filtered_df[filtered_df['vibration'] <= vib_threshold]['ID'],
        y=filtered_df[filtered_df['vibration'] <= vib_threshold]['vibration'],
        mode='markers',
        name='Normal Readings',
        marker=dict(color='#00d4ff', size=5, opacity=0.6),
        hovertemplate='<b>Normal</b><br>ID: %{x}<br>Vibration: %{y:.4f}<extra></extra>'
    ))
    
    # Add anomalies
    fig_anomaly.add_trace(go.Scatter(
        x=anomalies_df['ID'],
        y=anomalies_df['vibration'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='#ff4757', size=10, symbol='diamond', line=dict(color='#fff', width=2)),
        hovertemplate='<b>Anomaly</b><br>ID: %{x}<br>Vibration: %{y:.4f}<extra></extra>'
    ))
    
    # Add threshold line
    fig_anomaly.add_hline(
        y=vib_threshold,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Threshold: {vib_threshold:.4f}"
    )
    
    fig_anomaly.update_layout(
        title="Anomaly Detection Results",
        xaxis_title="Reading ID",
        yaxis_title="Vibration Level",
        height=450,
        plot_bgcolor='rgba(15, 39, 68, 0.3)',
        paper_bgcolor='rgba(15, 39, 68, 0.3)',
        font=dict(color='#e2e8f0'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Show top anomalies
    st.markdown("### 📋 Top 10 Anomalies by Vibration Level")
    top_anomalies = anomalies_df.nlargest(10, 'vibration')[['ID', 'vibration', 'revolutions', 'humidity']].reset_index(drop=True)
    st.dataframe(
        top_anomalies,
        use_container_width=True,
        column_config={
            'ID': st.column_config.NumberColumn('Reading ID', format='%d'),
            'vibration': st.column_config.NumberColumn('Vibration', format='%.4f'),
            'revolutions': st.column_config.NumberColumn('Revolutions', format='%.2f'),
            'humidity': st.column_config.NumberColumn('Humidity', format='%.2f%%')
        }
    )
else:
    st.success("✅ No anomalies detected above the threshold!")

# ============================================
# SECTION 8: INSIGHTS PANEL
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>💡 Automated Insights</div>
</div>
""", unsafe_allow_html=True)

# Generate dynamic insights based on filtered data
insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    # Correlation insights
    if abs(corr_coef) > 0.7:
        st.success(f"🔗 **Strong Correlation**: There's a strong {'positive' if corr_coef > 0 else 'negative'} correlation ({corr_coef:.3f}) between revolutions and vibration. Consider investigating mechanical linkage issues.")
    elif abs(corr_coef) > 0.4:
        st.info(f"📊 **Moderate Correlation**: A {'positive' if corr_coef > 0 else 'negative'} correlation ({corr_coef:.3f}) exists between revolutions and vibration. Monitor for potential wear patterns.")
    else:
        st.warning(f"⚠️ **Weak Correlation**: Low correlation ({corr_coef:.3f}) between revolutions and vibration suggests other factors may be influencing vibration levels.")
    
    # Anomaly rate insight
    if anomaly_percentage > 5:
        st.error(f"🚨 **High Anomaly Rate**: {anomaly_percentage:.2f}% of readings exceed the threshold. Immediate maintenance recommended!")
    elif anomaly_percentage > 2:
        st.warning(f"⚠️ **Elevated Anomaly Rate**: {anomaly_percentage:.2f}% of readings are anomalies. Schedule maintenance inspection.")
    else:
        st.success(f"✅ **Normal Operation**: Anomaly rate of {anomaly_percentage:.2f}% is within acceptable limits.")
    
    # Humidity impact insight
    hum_vib_corr = filtered_df['humidity'].corr(filtered_df['vibration'])
    if abs(hum_vib_corr) > 0.3:
        st.info(f"💧 **Humidity Influence**: Humidity shows a {'positive' if hum_vib_corr > 0 else 'negative'} correlation ({hum_vib_corr:.3f}) with vibration. Environmental factors may affect system performance.")
    else:
        st.success(f"🌡️ **Stable Environment**: Humidity impact on vibration is minimal ({hum_vib_corr:.3f}). System is well-insulated from environmental factors.")

with insights_col2:
    # Vibration level insight
    if avg_vib > df['vibration'].quantile(0.75):
        st.error(f"📈 **Elevated Vibration**: Current average vibration ({avg_vib:.4f}) is above the 75th percentile. Check for mechanical issues.")
    elif avg_vib > df['vibration'].median():
        st.warning(f"⚠️ **Moderate Vibration**: Vibration levels ({avg_vib:.4f}) are above median. Continue monitoring.")
    else:
        st.success(f"✅ **Optimal Vibration**: Vibration levels ({avg_vib:.4f}) are within normal range.")
    
    # Sensor health insight
    total_outliers = sum([info['Outliers'] for info in outlier_info])
    if total_outliers > len(filtered_df) * 0.05:
        st.error(f"🔬 **Sensor Issues**: Sensors are showing {total_outliers} outliers. Calibrate or replace sensors as needed.")
    elif total_outliers > len(filtered_df) * 0.02:
        st.warning(f"📊 **Sensor Variability**: {total_outliers} outlier readings detected across sensors. Consider sensor maintenance.")
    else:
        st.success(f"🔬 **Healthy Sensors**: All sensors are functioning within expected parameters.")
    
    # Revolutions insight
    if avg_rev > df['revolutions'].quantile(0.9):
        st.warning(f"🔄 **High RPM**: Average revolutions ({avg_rev:.2f}) are high. Monitor for wear and tear.")
    elif avg_rev < df['revolutions'].quantile(0.1):
        st.info(f"🔄 **Low RPM**: Average revolutions ({avg_rev:.2f}) are low. Verify motor efficiency.")
    else:
        st.success(f"✅ **Normal RPM**: Revolutions are within optimal operating range.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #64748b;'>
    <p>🛗 Smart Elevator Predictive Maintenance Dashboard</p>
    <p style='font-size: 0.85rem;'>Mathematics for AI Summative Assessment Project</p>
    <p style='font-size: 0.75rem;'>Built with Streamlit, Plotly, and Seaborn</p>
</div>
""", unsafe_allow_html=True)
