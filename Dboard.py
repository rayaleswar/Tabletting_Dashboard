import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('production_data_skewed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.to_period('W').astype(str)
    return df

df = load_data()

def create_separate_plots():
    # Downtime by Production Line
    downtime_by_line = df.groupby('Production Line')['Downtime (hrs)'].sum().reset_index()

    # Downtime Distribution Pie Chart
    fig1 = px.pie(downtime_by_line, 
                  names='Production Line', 
                  values='Downtime (hrs)', 
                  title='Downtime Distribution by Production Line',
                  color_discrete_sequence=px.colors.qualitative.Set3)

    # Defect Rate by Production Line
    defect_rate_by_line = df.groupby('Production Line')['Tablet Defect Rate (%)'].mean().reset_index()

    # Tablet Defect Rate Bar Chart
    fig2 = px.bar(defect_rate_by_line, 
                  x='Production Line', 
                  y='Tablet Defect Rate (%)', 
                  color='Production Line',  
                  title='Tablet Defect Rate by Production Line')

    # Cycle Count by Production Line
    cycle_count_by_line = df.groupby('Production Line')['Cycle Count'].mean().reset_index()

    # Average Cycle Count Bar Chart
    fig3 = px.bar(cycle_count_by_line, 
                  x='Production Line', 
                  y='Cycle Count', 
                  color='Production Line', 
                  title='Average Cycle Count by Production Line')

    # Energy Consumption by Production Line
    energy_consumption_by_line = df.groupby('Production Line')['Energy Consumption (kWh)'].mean().reset_index()

    # Energy Consumption Bar Chart
    fig4 = px.bar(energy_consumption_by_line, 
                  x='Production Line', 
                  y='Energy Consumption (kWh)', 
                  color='Production Line',  
                  title='Energy Consumption by Production Line')

    return [fig1, fig2, fig3, fig4]

# Streamlit app
st.title("Tablet Manufacturing Dashboard")

# Create charts
charts = create_separate_plots()

# Create a 3-column layout
col1, col2, col3 = st.columns([2, 2, 4])  # Increase col3 width to span more screen space

# Left column (2 charts)
with col1:
    st.plotly_chart(charts[0], use_container_width=True)  # Downtime Distribution Pie
    st.plotly_chart(charts[1], use_container_width=True)  # Tablet Defect Rate Bar

# Middle column (2 charts)
with col2:
    st.plotly_chart(charts[2], use_container_width=True)  # Average Cycle Count Bar
    st.plotly_chart(charts[3], use_container_width=True)  # Energy Consumption Bar

# Right column: 3 Separate Line Plots for Each Production Line

# Filter data by Production Line
def create_vibration_line_plot(line_data, title):
    # Calculate mean, Q1, and Q3
    mean = line_data['Vibration Level (mm/s)'].mean()
    q1 = line_data['Vibration Level (mm/s)'].quantile(0.25) / 2
    q3 = line_data['Vibration Level (mm/s)'].quantile(0.75) * 1.3
    
    # Create the line plot
    fig = px.line(line_data, 
                  x='Week', 
                  y='Vibration Level (mm/s)', 
                  title=title, 
                  line_shape='spline',  
                  render_mode='svg')
    
    # Add mean line
    fig.add_shape(type='line',
                  x0=line_data['Week'].min(),
                  x1=line_data['Week'].max(),
                  y0=mean,
                  y1=mean,
                  line=dict(color='red', width=2, dash='dash'),
                  name='Mean Vibration Level')
    
    # Add Q1 line
    fig.add_shape(type='line',
                  x0=line_data['Week'].min(),
                  x1=line_data['Week'].max(),
                  y0=q1,
                  y1=q1,
                  line=dict(color='orange', width=2, dash='dash'),
                  name='Lower Control Limit (Q1)')
    
    # Add Q3 line
    fig.add_shape(type='line',
                  x0=line_data['Week'].min(),
                  x1=line_data['Week'].max(),
                  y0=q3,
                  y1=q3,
                  line=dict(color='green', width=2, dash='dash'),
                  name='Upper Control Limit (Q3)')

    # Customize the layout
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Avg Vibration Level (mm/s)',
        legend_title=None,
        font=dict(size=12),
        width = 500,
        height = 300,
        hovermode='x unified',
        showlegend=False
    )
    
    # Remove x-axis tick labels
    fig.update_xaxes(showticklabels=False)

    return fig

# Create individual plots for each production line
line_1_data = df[df['Production Line'] == 'Line 1']
line_2_data = df[df['Production Line'] == 'Line 2']
line_3_data = df[df['Production Line'] == 'Line 3']

# Generate the vibration level line plots
fig5_line_1 = create_vibration_line_plot(line_1_data, 'Vibration Level (Line 1)')
fig5_line_2 = create_vibration_line_plot(line_2_data, 'Vibration Level (Line 2)')
fig5_line_3 = create_vibration_line_plot(line_3_data, 'Vibration Level (Line 3)')

# Display the 3 separate line plots vertically in the right column
with col3:
    st.plotly_chart(fig5_line_1, use_container_width=True)
    st.plotly_chart(fig5_line_2, use_container_width=True)
    st.plotly_chart(fig5_line_3, use_container_width=True)

def meantimebwfailures():
    df['Date'] = pd.to_datetime(df['Date'])
    # Define vibration threshold for failure
    vibration_threshold = 6  # This can be adjusted based on your requirements

    # Filter relevant columns for analysis
    weekly_data = df[['Date', 'Vibration Level (mm/s)', 'Production Line', 'Machine Runtime (hrs)']]

    # Calculate MTBF (Mean Time Between Failures) for each production line
    production_lines = weekly_data['Production Line'].unique()
    mtbf_data = []

    for line in production_lines:
        line_data = weekly_data[weekly_data['Production Line'] == line]
        # Determine failure dates based on vibration threshold
        failure_dates = line_data[line_data['Vibration Level (mm/s)'] > vibration_threshold]['Date']
        # Calculate the difference between failure dates
        mtbf = failure_dates.diff().dt.total_seconds() / 3600  # Convert difference to hours
        line_data = line_data.copy()  # Create a copy to avoid modifying original data
        line_data['MTBF (hrs)'] = mtbf.fillna(0)  # Fill NA values with 0 for initial entries
        mtbf_data.append(line_data)

    # Combine data for all production lines
    mtbf_combined = pd.concat(mtbf_data)

    # Plot MTBF over time for each production line
    fig = px.line(mtbf_combined, 
                x='Date', 
                y='MTBF (hrs)', 
                color='Production Line',  # Differentiate by production line
                title='Mean Time Between Failures (MTBF) by Production Line',
                labels={'Date': 'Date', 'MTBF (hrs)': 'Mean Time Between Failures (Hours)'},
                line_shape='linear')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='MTBF (Hours)',
        width=800,
        height=800,
        font=dict(size=12),
        hovermode='x unified'
    )

    return fig

# st.plotly_chart(meantimebwfailures(), use_container_width=True)

def correlationplot():
    data = df[['Downtime (hrs)', 'Bearing Health Score (%)', 'Lubrication Level (%)',\
            'Vibration Level (mm/s)', 'Energy Consumption (kWh)', 'Machine Runtime (hrs)','Temperature (°C)']]

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Create a heatmap with Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        labels=dict(x="Metrics", y="Metrics", color="Correlation"),
        x=data.columns,
        y=data.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )

    # Remove x-axis ticks
    fig.update_xaxes(showticklabels=False)

    # Update layout for better readability
    fig.update_layout(
        title="Correlation Heatmap Between Machine Metrics",
        width=800,
        height=800,
        xaxis_title="",
        yaxis_title="",
        font=dict(size=12)
    )
    return fig

# st.plotly_chart(correlationplot(), use_container_width=True)

r2col1, r2col2 = st.columns([1, 1])  # Set equal width for both columns

with r2col1:
    st.plotly_chart(meantimebwfailures(), use_container_width=True)  # MTBF plot

with r2col2:
    st.plotly_chart(correlationplot(), use_container_width=True)  # Correlation plot

def scatterplot3d():
    # Create a 3D scatter plot
    fig = px.scatter_3d(df, 
                        x='Vibration Level (mm/s)', 
                        y='Temperature (°C)', 
                        z='Energy Consumption (kWh)', 
                        color='Production Line',  # Different colors for each production line
                        size='Cycle Count',  # Optionally, size points by cycle count
                        hover_data=['Date'],  # Add date to hover info
                        title='Vibration Level, Temperature, and Energy Consumption')

    fig.update_layout(xaxis_title='Week Starting',
                        yaxis_title='Average Vibration Level (mm/s)',
                        legend_title='Production Line',
                        font=dict(size=12),
                        width = 1000,
                        height = 1000,
                        hovermode='x unified'
                    ) 
    return fig

st.plotly_chart(scatterplot3d(), use_container_width=True)