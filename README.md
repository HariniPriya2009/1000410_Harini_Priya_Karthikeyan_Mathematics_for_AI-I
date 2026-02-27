# 1000410_Harini_Priya_Karthikeyan_Math_in_AI

Smart Elevator Predictive Maintenance Dashboard

A production-ready Streamlit application for real-time elevator system monitoring and predictive maintenance analysis.

App live link: https://1000410-harini-priya-karthikeyan-mathematics-for-ai.streamlit.app/

Canva Story Board link: https://www.canva.com/design/DAHCUVu_wMA/i8dsdvuy-J02t3VXm6gFMA/view?utm_content=DAHCUVu_wMA&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h28b4fcdae3

What Does This Web App Visualize?

This dashboard provides comprehensive visualization and analysis of elevator system sensor data to enable predictive maintenance. It visualizes:

Key Visualizations:

- Time Series Analysis (Line Plot)

- Vibration levels over time (ID vs vibration)

- Identifies unusual spikes and anomalies

- Shows upper and lower threshold limits

- Distribution Analysis (Histograms)

- Humidity distribution across readings

- Revolutions distribution patterns

- Identifies normal ranges vs extreme values

- Scatter Plot Analysis

- Relationship between revolutions and vibration

- Trend line showing correlation direction

- Statistical metrics (correlation coefficient, R-squared)

- Sensor Health Check (Box Plot)

- Distribution of sensor readings (x1, x2, x3, x4, x5)

- Outlier detection for each sensor

- Comparative analysis across sensors

- Correlation Heatmap

- Relationships between all numeric variables

- Identifies how humidity affects vibration

- Shows interdependencies between sensors and operational parameters

- Anomaly Detection

- Real-time anomaly identification

- Threshold-based alerting system

- Top anomalies ranking and analysis

- Automated Insights

- Dynamic insights based on filtered data

- Maintenance recommendations

- System health status indicators

Project Overview

Purpose

The Smart Elevator Predictive Maintenance Dashboard is designed to monitor elevator system health in real-time, detect potential issues before they become critical, and optimize maintenance schedules through data-driven insights.

Key Features

- Real-time Monitoring: Live dashboard with interactive visualizations

- Predictive Analytics: Anomaly detection and trend analysis

- Interactive Filtering: Adjustable parameters for focused analysis

- Automated Insights: AI-powered recommendations

- Responsive Design: Modern UI with neon purple theme

- Comprehensive Analysis: Multiple visualization types for complete system understanding

Target Users

- Elevator maintenance technicians

- Building facility managers

- Predictive maintenance engineers

- Data analysts in the building management sector

🔧 Integration Details

Technology Stack

Frontend Framework:

- Streamlit (v1.28.0+): Python web application framework

- Plotly: Interactive plotting library

- Seaborn: Statistical data visualization

- Matplotlib: Foundation plotting library

Data Processing:

- Pandas: Data manipulation and analysis

- NumPy: Numerical computing

- SciPy: Statistical analysis and scientific computing

Styling:

- Custom CSS with neon purple theme

- Responsive design principles

- Dark mode optimized interface

Data Requirements

Required Dataset:

- File: Elevator predictive-maintenance-dataset.csv

- Format: CSV (Comma Separated Values)

Required Columns:

- ID: Unique identifier for each reading (time/sample index)

- revolutions: Motor revolutions per reading

- humidity: Environmental humidity percentage

- vibration: Vibration level readings

- x1, x2, x3, x4, x5: Additional sensor readings

Data Processing:

- Automatic duplicate removal

- Missing value interpolation

- Data type validation

- Outlier detection and handling

Modules Used

streamlit

pandas

numpy

plotly

seaborn

matplotlib

scipy

Deployment Instructions

- GitHub account

- Streamlit Cloud account (free tier )

- Repository with my code

Step 1: Prepare Repository

- Create a GitHub repository

- Upload the following files:

- app.py (or app_updated.py)

- Elevator predictive-maintenance-dataset.csv

- requirements.txt

- README.md

Step 2: Create requirements.txt

Create a requirements.txt file with:

streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
seaborn>=0.12.0
matplotlib>=3.7.0
scipy>=1.11.0

Step 3: Deploy to Streamlit Cloud

- Go to share.streamlit.io

- Click "New app"

- Connect your GitHub account

- Select your repository

- Choose the main branch

- Set main file path: app.py

- Click "Deploy"

Step 4: Access Your App

Once deployed, Streamlit Cloud will provide a live URL for your dashboard.

Alternative Deployment Options

Docker Deployment

FROM python:3.11-slim

Live Web App/Dashboard Streamlit Cloud Link

https://1000410-harini-priya-karthikeyan-mathematics-for-ai.streamlit.app/


Deployment Status

- ✅ Code ready for deployment

- ✅ All dependencies documented

- ✅ Dataset integration complete


📱 Dashboard Features

Interactive Controls

- Revolutions Range Slider: Filter data by revolution values

- Humidity Range Slider: Filter by humidity percentage

- Vibration Threshold: Set custom anomaly detection threshold

- Reset Filters: Quickly return to default view

Key Performance Indicators

- Average Vibration

- Maximum Vibration

- Average Revolutions

- Average Humidity

Visual Analytics

- Time series plots with anomaly highlighting

- Distribution histograms

- Scatter plots with trend analysis

- Box plots for outlier detection

- Correlation heatmaps

- Anomaly detection visualizations

Automated Insights

- Real-time anomaly rate monitoring

- Environmental impact analysis

- Sensor health diagnostics

- Maintenance recommendations



Theme 

The dashboard uses a neon purple theme. To customize colors, modify the CSS section in app.py:

📊 Data Privacy & Security

- All data processing happens locally

- No external API calls

- Suitable for sensitive operational data

Credits

Created by: Harini Priya Karthikeyan (ID: 1000410)

CRS: Artificial Intelligence: 

Course Name: Mathematics for AI-I

Mentor: Syed Ali Beema.S

School: Jain Vidyalaya IB world school, Madurai
