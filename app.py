"""
Smart Elevator Predictive Maintenance Dashboard
A production-ready Streamlit application for elevator system monitoring
NEON PURPLE THEME VERSION
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

st.markdown("""
<style>

/* ===============================
   MAIN BACKGROUND
=================================*/
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a0b2e 40%, #240046 100%);
    color: #e0e0ff;
}

/* ===============================
   HEADER
=================================*/
.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    color: #b388ff;
    text-shadow: 0 0 20px #8e2de2, 0 0 40px #4a00e0;
}

.header-subtitle {
    text-align: center;
    color: #c77dff;
    margin-bottom: 2rem;
    font-weight: 500;
}

/* ===============================
   SECTION CONTAINERS
=================================*/
.section-container {
    background: rgba(30, 0, 60, 0.7);
    padding: 1.5rem;
    border-radius: 18px;
    margin-bottom: 2rem;
    border: 1px solid #8e2de2;
    box-shadow: 0 0 25px rgba(142, 45, 226, 0.5);
}

/* ===============================
   SECTION TITLES
=================================*/
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #b388ff;
    text-shadow: 0 0 10px #8e2de2;
}

/* ===============================
   METRIC CARDS
=================================*/
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1a0033 0%, #2d0066 100%);
    border-radius: 14px;
    padding: 15px;
    border: 1px solid #8e2de2;
    box-shadow: 0 0 15px rgba(142, 45, 226, 0.4);
}

div[data-testid="stMetricLabel"] {
    color: #c77dff;
}

div[data-testid="stMetricValue"] {
    color: #00f0ff;
    text-shadow: 0 0 10px #00f0ff;
}

/* ===============================
   CUSTOM COLOR CLASSES FOR METRIC LABELS
=================================*/
.custom-metric-label {
    color: #00f0ff !important;
    font-weight: 600;
    text-shadow: 0 0 8px #00f0ff;
}

/* ===============================
   CUSTOM COLOR CLASSES FOR SIDEBAR HEADINGS
=================================*/
.custom-sidebar-heading {
    color: #00f0ff !important;
    font-weight: 600;
    text-shadow: 0 0 8px #00f0ff;
}

/* ===============================
   SIDEBAR
=================================*/
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0033 0%, #0f001f 100%);
}

/* ===============================
   BUTTONS
=================================*/
.stButton>button {
    background: linear-gradient(90deg, #8e2de2, #4a00e0);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #c77dff, #7209b7);
    box-shadow: 0 0 15px #8e2de2;
}

/* ===============================
   ALERT BOXES
=================================*/
.stInfo {
    background-color: rgba(0, 240, 255, 0.15) !important;
    border-left: 4px solid #00f0ff !important;
}

.stSuccess {
    background-color: rgba(67, 233, 123, 0.15) !important;
    border-left: 4px solid #43e97b !important;
}

.stWarning {
    background-color: rgba(255, 170, 0, 0.15) !important;
    border-left: 4px solid #ffaa00 !important;
}

.stError {
    background-color: rgba(255, 71, 87, 0.15) !important;
    border-left: 4px solid #ff4757 !important;
}

/* ===============================
   REMOVE STREAMLIT BRANDING
=================================*/
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

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
<div style='padding: 1rem; background: rgba(142, 45, 226, 0.2); border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #8e2de2;'>
    <h3 style='color: #b388ff; margin: 0;'>🎛️ Control Panel</h3>
    <p style='color: #c77dff; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Adjust filters to analyze specific data ranges</p>
</div>
""", unsafe_allow_html=True)

# Revolutions range slider
st.sidebar.markdown("<p class='custom-sidebar-heading'>🔄 Revolutions Range</p>", unsafe_allow_html=True)
min_rev, max_rev = float(df['revolutions'].min()), float(df['revolutions'].max())
rev_range = st.sidebar.slider(
    "",
    min_value=min_rev,
    max_value=max_rev,
    value=(min_rev, max_rev),
    step=10.0,
    help="Filter data based on revolutions"
)

# Humidity range slider
st.sidebar.markdown("<p class='custom-sidebar-heading'>💧 Humidity Range (%)</p>", unsafe_allow_html=True)
min_hum, max_hum = float(df['humidity'].min()), float(df['humidity'].max())
hum_range = st.sidebar.slider(
    "",
    min_value=min_hum,
    max_value=max_hum,
    value=(min_hum, max_hum),
    step=1.0,
    help="Filter data based on humidity levels"
)

# Vibration threshold slider
st.sidebar.markdown("<p class='custom-sidebar-heading'>📊 Vibration Threshold</p>", unsafe_allow_html=True)
vib_threshold = st.sidebar.slider(
    "",
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
    st.markdown("<p class='custom-metric-label'>Average Vibration</p>", unsafe_allow_html=True)
    st.metric(
        label="",
        value=f"{avg_vib:.4f}",
        delta=f"{(avg_vib/df['vibration'].mean()*100)-100:.1f}%",
        help="Mean vibration level across all readings"
    )

with col2:
    max_vib = filtered_df['vibration'].max()
    st.markdown("<p class='custom-metric-label'>Max Vibration</p>", unsafe_allow_html=True)
    st.metric(
        label="",
        value=f"{max_vib:.4f}",
        delta=f"{(max_vib/df['vibration'].max()*100)-100:.1f}%",
        help="Maximum vibration reading"
    )

with col3:
    avg_rev = filtered_df['revolutions'].mean()
    st.markdown("<p class='custom-metric-label'>Avg Revolutions</p>", unsafe_allow_html=True)
    st.metric(
        label="",
        value=f"{avg_rev:.2f}",
        delta=f"{(avg_rev/df['revolutions'].mean()*100)-100:.1f}%",
        help="Average revolutions per reading"
    )

with col4:
    avg_hum = filtered_df['humidity'].mean()
    st.markdown("<p class='custom-metric-label'>Humidity</p>", unsafe_allow_html=True)
    st.metric(
        label="",
        value=f"{avg_hum:.2f}%",
        delta=f"{(avg_hum/df['humidity'].mean()*100)-100:.1f}%",
        help="Average humidity percentage"
    )

# ============================================
# SECTION 2: TIME SERIES ANALYSIS (Line Plot)
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>⏱️ Time Series Analysis - Vibration Over Time</div>
</div>
""", unsafe_allow_html=True)

# Calculate statistics for anomaly highlighting
vib_mean = filtered_df['vibration'].mean()
vib_std = filtered_df['vibration'].std()
vib_upper = vib_mean + 2 * vib_std
vib_lower = vib_mean - 2 * vib_std

# Create time series plot
fig_ts = go.Figure()

# Add main line - PURPLE
fig_ts.add_trace(go.Scatter(
    x=filtered_df['ID'],
    y=filtered_df['vibration'],
    mode='lines',
    name='Vibration',
    line=dict(color='#b388ff', width=1.5),
    hovertemplate='<b>ID: %{x}</b><br>Vibration: %{y:.4f}<extra></extra>'
))

# Highlight abnormal spikes - PINK
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
            color='#ff6b9d',
            size=8,
            symbol='diamond',
            line=dict(color='#fff', width=1)
        ),
        hovertemplate='<b>Anomaly Detected</b><br>ID: %{x}<br>Vibration: %{y:.4f}<extra></extra>'
    ))

# Add threshold lines - MAGENTA
fig_ts.add_hline(y=vib_upper, line_dash="dash", line_color="#c77dff", annotation_text="Upper Limit")
fig_ts.add_hline(y=vib_lower, line_dash="dash", line_color="#c77dff", annotation_text="Lower Limit")

fig_ts.update_layout(
    title=dict(
        text="Vibration Levels Over Time",
        font=dict(size=18, color='#00f0ff', family='Arial Black')
    ),
    xaxis=dict(
        title=dict(
            text="Reading ID",
            font=dict(size=14, color='#c77dff', family='Arial')
        ),
        tickfont=dict(size=12, color='#b388ff')
    ),
    yaxis=dict(
        title=dict(
            text="Vibration Level",
            font=dict(size=14, color='#c77dff', family='Arial')
        ),
        tickfont=dict(size=12, color='#b388ff')
    ),
    hovermode='x unified',
    template='plotly_dark',
    height=400,
    plot_bgcolor='rgba(15, 12, 41, 0.3)',
    paper_bgcolor='rgba(15, 12, 41, 0.3)',
    font=dict(color='#b388ff'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12, color='#b388ff')
    )
)

st.plotly_chart(fig_ts, use_container_width=True)

# ============================================
# SECTION 3: DISTRIBUTION ANALYSIS (Histograms)
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>📊 Distribution Analysis - Humidity & Revolutions</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_rev_dist = px.histogram(
        filtered_df,
        x='revolutions',
        nbins=50,
        title='Distribution of Revolutions',
        color_discrete_sequence=['#c77dff'],  # Light Purple
        template='plotly_dark'
    )
    
    fig_rev_dist.update_layout(
        title=dict(
            text='Distribution of Revolutions',
            font=dict(size=16, color='#00f0ff', family='Arial Black')
        ),
        xaxis=dict(
            title=dict(
                text="Revolutions",
                font=dict(size=14, color='#c77dff', family='Arial')
            ),
            tickfont=dict(size=12, color='#b388ff')
        ),
        yaxis=dict(
            title=dict(
                text="Count",
                font=dict(size=14, color='#c77dff', family='Arial')
            ),
            tickfont=dict(size=12, color='#b388ff')
        ),
        height=350,
        plot_bgcolor='rgba(15, 12, 41, 0.3)',
        paper_bgcolor='rgba(15, 12, 41, 0.3)',
        font=dict(color='#b388ff')
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
        color_discrete_sequence=['#b388ff'],  # Purple
        template='plotly_dark'
    )
    
    fig_hum_dist.update_layout(
        title=dict(
            text='Distribution of Humidity',
            font=dict(size=16, color='#00f0ff', family='Arial Black')
        ),
        xaxis=dict(
            title=dict(
                text="Humidity (%)",
                font=dict(size=14, color='#c77dff', family='Arial')
            ),
            tickfont=dict(size=12, color='#b388ff')
        ),
        yaxis=dict(
            title=dict(
                text="Count",
                font=dict(size=14, color='#c77dff', family='Arial')
            ),
            tickfont=dict(size=12, color='#b388ff')
        ),
        height=350,
        plot_bgcolor='rgba(15, 12, 41, 0.3)',
        paper_bgcolor='rgba(15, 12, 41, 0.3)',
        font=dict(color='#b388ff')
    )
    
    fig_hum_dist.update_traces(
        hovertemplate='<b>Humidity: %{x:.2f}%</b><br>Count: %{y}<extra></extra>'
    )
    
    st.plotly_chart(fig_hum_dist, use_container_width=True)

# ============================================
# SECTION 4: BOX PLOT (Sensor Readings x1-x5)
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🔬 Sensor Health Check - Box Plot Analysis</div>
</div>
""", unsafe_allow_html=True)

# Prepare sensor data
sensor_columns = ['x1', 'x2', 'x3', 'x4', 'x5']
sensor_data = filtered_df[sensor_columns].melt(var_name='Sensor', value_name='Reading')

# Create box plot - Purple gradient palette
fig, ax = plt.subplots(figsize=(12, 6))
custom_palette = ['#b388ff', '#c77dff', '#8e2de2', '#7209b7', '#4a00e0']  # Purple shades
sns.boxplot(
    data=sensor_data,
    x='Sensor',
    y='Reading',
    palette=custom_palette,
    ax=ax,
    showfliers=True
)

# Customize plot
ax.set_xlabel('Sensor ID', fontsize=12, color='#b388ff')
ax.set_ylabel('Reading Value', fontsize=12, color='#b388ff')
ax.set_title('Sensor Reading Distribution with Outliers', fontsize=14, color='#b388ff', pad=20)

# Style the plot
ax.spines['bottom'].set_color('#8e2de2')
ax.spines['top'].set_color('#8e2de2')
ax.spines['left'].set_color('#8e2de2')
ax.spines['right'].set_color('#8e2de2')
ax.tick_params(axis='x', colors='#b388ff')
ax.tick_params(axis='y', colors='#b388ff')
ax.xaxis.label.set_color('#b388ff')
ax.yaxis.label.set_color('#b388ff')
ax.title.set_color('#b388ff')

# Set background
ax.set_facecolor('#0f0c29')
fig.patch.set_facecolor('#0a001a')

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
# SECTION 5: CORRELATION HEATMAP
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🌡️ Correlation Heatmap - Feature Relationships</div>
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
    cmap='Purples',  # Purple colormap
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={'shrink': 0.8},
    ax=ax,
    annot_kws={'size': 10, 'color': 'white', 'weight': 'bold'}
)

# Customize plot
ax.set_title('Feature Correlation Matrix', fontsize=16, color='#b388ff', pad=20)
ax.tick_params(axis='x', colors='#b388ff', rotation=45)
ax.tick_params(axis='y', colors='#b388ff', rotation=0)
ax.set_facecolor('#1a0033')
fig.patch.set_facecolor('#0f0c29')

plt.tight_layout()
st.pyplot(fig_heatmap, use_container_width=True)

# ============================================
# SECTION 6: ANOMALY DETECTION
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>🚨 Anomaly Detection - Vibration Analysis</div>
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
    
    # Add normal readings - LIGHT PURPLE
    fig_anomaly.add_trace(go.Scatter(
        x=filtered_df[filtered_df['vibration'] <= vib_threshold]['ID'],
        y=filtered_df[filtered_df['vibration'] <= vib_threshold]['vibration'],
        mode='markers',
        name='Normal Readings',
        marker=dict(color='#c77dff', size=5, opacity=0.6),
        hovertemplate='<b>Normal</b><br>ID: %{x}<br>Vibration: %{y:.4f}<extra></extra>'
    ))
    
    # Add anomalies - PINK
    fig_anomaly.add_trace(go.Scatter(
        x=anomalies_df['ID'],
        y=anomalies_df['vibration'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='#ff6b9d', size=10, symbol='diamond', line=dict(color='#fff', width=2)),
        hovertemplate='<b>Anomaly</b><br>ID: %{x}<br>Vibration: %{y:.4f}<extra></extra>'
    ))
    
    # Add threshold line
    fig_anomaly.add_hline(
        y=vib_threshold,
        line_dash="dash",
        line_color="#8e2de2",
        line_width=2,
        annotation_text=f"Threshold: {vib_threshold:.4f}"
    )
    
    fig_anomaly.update_layout(
        title=dict(
            text="Anomaly Detection Results",
            font=dict(size=18, color='#00f0ff', family='Arial Black')
        ),
        xaxis=dict(
            title=dict(
                text="Reading ID",
                font=dict(size=14, color='#c77dff', family='Arial')
            ),
            tickfont=dict(size=12, color='#b388ff')
        ),
        yaxis=dict(
            title=dict(
                text="Vibration Level",
                font=dict(size=14, color='#c77dff', family='Arial')
            ),
            tickfont=dict(size=12, color='#b388ff')
        ),
        height=450,
        plot_bgcolor='rgba(15, 12, 41, 0.3)',
        paper_bgcolor='rgba(15, 12, 41, 0.3)',
        font=dict(color='#b388ff'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#b388ff')
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
# SECTION 7: INSIGHTS PANEL
# ============================================
st.markdown("""
<div class='section-container'>
    <div class='section-title'>💡 Automated Insights</div>
</div>
""", unsafe_allow_html=True)

# Generate dynamic insights based on filtered data
insights_col1, insights_col2 = st.columns(2)

with insights_col1:
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
    
    # Vibration level insight
    if avg_vib > df['vibration'].quantile(0.75):
        st.error(f"📈 **Elevated Vibration**: Current average vibration ({avg_vib:.4f}) is above the 75th percentile. Check for mechanical issues.")
    elif avg_vib > df['vibration'].median():
        st.warning(f"⚠️ **Moderate Vibration**: Vibration levels ({avg_vib:.4f}) are above median. Continue monitoring.")
    else:
        st.success(f"✅ **Optimal Vibration**: Vibration levels ({avg_vib:.4f}) are within normal range.")

with insights_col2:
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
    
    # Correlation insight
    rev_vib_corr = filtered_df['revolutions'].corr(filtered_df['vibration'])
    if abs(rev_vib_corr) > 0.5:
        st.info(f"🔗 **Strong Relationship**: Revolutions and vibration show a {'positive' if rev_vib_corr > 0 else 'negative'} correlation ({rev_vib_corr:.3f}). Monitor door usage patterns.")
    else:
        st.success(f"✅ **Independent Factors**: Revolutions and vibration are weakly correlated ({rev_vib_corr:.3f}). Other factors may influence vibration levels.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #b388ff;'>
    <p>🛗 Smart Elevator Predictive Maintenance Dashboard</p>
    <p style='font-size: 0.85rem; color: #c77dff;'>Mathematics for AI Summative Assessment Project</p>
    <p style='font-size: 0.75rem; color: #8e2de2;'>Built with Streamlit, Plotly, and Seaborn</p>
</div>
""", unsafe_allow_html=True)
