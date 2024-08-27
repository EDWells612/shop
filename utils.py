import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import ast
from mlxtend.frequent_patterns import apriori, association_rules
import re
def get_list_items(text):
    return re.findall(r'\d+\s([a-zA-Z-]+(?:\s[a-zA-Z-]+)*)', text)

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

def get_items(data):
    items = data['items'].apply(parse_items).explode().unique()
    items = [item for item in items if isinstance(item, str)]
    return sorted(items)

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
def get_items_input(items):
    items = items
    st.session_state.new_items = st.session_state.get('new_items', [])
    res = ""
    column1, column2, column3 = st.columns(3)
    with column1:
        item = st.selectbox('Select Item', ["new Item"] + items, key='item', index=None)
        if item == "new Item":
            item = st.text_input('Item', "")
    with column2:
        amount = st.number_input('Amount', 1)
    with column3:
        add = st.button('Add', )
    if add:
        if item:
            st.session_state.new_items.append((item, amount))
    res = ', '.join([f'{amount} {item}' for item, amount in st.session_state.new_items])
    st.write(res)
    return res

def sum_item_amounts(df, column_name):
    item_sums = {}
    for row in df[column_name]:
        pairs = row.split(', ')
        for pair in pairs:
            if pair == '':
                continue
            amount, item = pair.split(' ', 1)
            amount = int(amount)
            if item in item_sums:
                item_sums[item] += amount
            else:
                item_sums[item] = amount
    result_df = pd.DataFrame(list(item_sums.items()), columns=['Item', 'Total Amount']).set_index(column_name)
    
    return result_df
