import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
data = pd.read_csv('shop.csv', parse_dates=['date'])

# Clean the Item column
def parse_items(item_str):
    if pd.isna(item_str) or item_str == '':
        return []
    try:
        item_list = ast.literal_eval(item_str)
        if isinstance(item_list, list):
            # Convert items to lowercase and handle specific value changes
            item_list = [i.strip().lower() for i in item_list if isinstance(i, str)]
            return item_list
    except (ValueError, SyntaxError):
        return []
    return []

# Define functions for plotting
def plot_items_in_channel(data, channel):
    filtered_channel_data = data[data['Channel'] == channel]
    items_list = filtered_channel_data['items'].apply(parse_items).explode()
    items_count = items_list.value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    items_count.plot(kind='barh', ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Items')
    ax.set_title(f'Items bought in {channel}')
    st.pyplot(fig)

def plot_channels_pie(data):
    channel_counts = data['Channel'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', textprops={'fontsize': 5})

    st.pyplot(fig)

def plot_item_distribution(data, item):
    filtered_data = data[data['items'].apply(lambda x: item in parse_items(x))]
    item_distribution = filtered_data['Channel'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    item_distribution.plot(kind='bar', ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Channels')
    ax.set_title(f'Distribution of {item}')
    st.pyplot(fig)

def perform_apriori_analysis(data, min_support=0.1, metric="lift", min_threshold=1.0):
    # Prepare the data for Apriori
    data['items_list'] = data['items'].apply(parse_items)
    oht = data['items_list'].str.join('|').str.get_dummies()
    
    # Apply Apriori
    frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    # Convert frozenset to strings for better readability
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    return rules


# Navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Channels", "Items"])

if page == "Home":
    st.title("Farahy Shop Analysis")
    st.write("Pie chart showing the distribution of channels.")
    plot_channels_pie(data)

elif page == "Channels":
    # Sidebar filters
    channel = st.sidebar.selectbox('Select Channel', data['Channel'].unique())
    date_range = st.sidebar.date_input(
        'Date range', [data['date'].min().date(), data['date'].max().date()])

    # Convert date_range to datetime objects
    start_date, end_date = pd.to_datetime(date_range)

    # Filter data
    filtered_data = data[(data['Channel'] == channel) &
                         (data['date'].between(start_date, end_date))]

    # Check if filtered_data is not empty
    if not filtered_data.empty:
        # Set Date column as index
        filtered_data.set_index('date', inplace=True)

        # Group by date
        time_series = filtered_data.resample('M').sum()

        # Plotting time series
        st.line_chart(time_series['Total Amount'])

        # Additional analysis
        st.write('Summary statistics for ', channel)
        st.write(filtered_data.describe())

        # Plot bar chart for items bought in the selected channel
        st.write(f'Items bought in {channel}')
        plot_items_in_channel(filtered_data, channel)

        # Apriori Analysis
        st.write(f"Apriori analysis for {channel}")
        rules = perform_apriori_analysis(filtered_data, min_support=0.1, metric="lift", min_threshold=1.0)
        
        if not rules.empty:
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.write("No significant rules found with the selected parameters.")

    else:
        st.write("No data available for the selected filters.")

elif page == "Items":
    st.title("Items Analysis")
    item = st.sidebar.selectbox('Select Item', pd.Series(data['items'].apply(parse_items).explode()).unique())

    if item:
        st.write(f'Distribution of {item}')
        plot_item_distribution(data, item)
