import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('shop.csv', parse_dates=['date'])

def parse_items(item_str):
    if pd.isna(item_str) or item_str == '':
        return []
    try:
        item_list = ast.literal_eval(item_str)
        if isinstance(item_list, list):
            item_list = [i.strip().lower() for i in item_list if isinstance(i, str)]
            return item_list
    except (ValueError, SyntaxError):
        return []
    return []

def plot_items_in_channel(data, channel):
    filtered_channel_data = data[data['Channel'] == channel] if channel != 'All' else data
    items_list = filtered_channel_data['items'].apply(parse_items).explode()
    items_count = items_list.value_counts()

    if items_count.empty:
        st.write(f"No items found in {channel}")
        return
    if channel != 'All':
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 20))
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
    data['items_list'] = data['items'].apply(parse_items)
    oht = data['items_list'].str.join('|').str.get_dummies()
    
    frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    return rules

page = st.sidebar.selectbox("Select Page", ["Home", "Channels", "Items"])

if page == "Home":
    st.title("Farahy Shop Analysis")
    st.write("Pie chart showing the distribution of channels.")
    plot_channels_pie(data)

elif page == "Channels":
    channel = st.sidebar.selectbox('Select Channel', ['All'] + list(data['Channel'].unique()))
    
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()
    
    start_date = st.sidebar.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input('End Date', min_value=start_date, max_value=max_date, value=max_date)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if channel == 'All':
        filtered_data = data[(data['date'].between(start_date, end_date))]
    else:
        filtered_data = data[(data['Channel'] == channel) &
                             (data['date'].between(start_date, end_date))]

    if not filtered_data.empty:
        filtered_data.set_index('date', inplace=True)

        time_series = filtered_data.resample('M').sum()

        st.line_chart(time_series['Total Amount'])

        st.write('Summary statistics for ', 'All' if channel == 'All' else channel)
        st.write(filtered_data.describe())

        st.write(f'Items bought in {"all channels" if channel == "All" else channel}')
        plot_items_in_channel(filtered_data, channel)

        st.write(f"Apriori analysis for {'all channels' if channel == 'All' else channel}")
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
    
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()
    
    start_date = st.sidebar.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input('End Date', min_value=start_date, max_value=max_date, value=max_date)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    filtered_data = data[(data['date'].between(start_date, end_date)) & 
                         (data['items'].apply(lambda x: item in parse_items(x)))]

    if item and not filtered_data.empty:
        st.write(f'Distribution of {item}')
        plot_item_distribution(filtered_data, item)
    else:
        st.write("No data available for the selected filters.")
