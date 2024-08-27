import streamlit as st
import pandas as pd
from utils import parse_items, plot_items_in_channel, plot_item_distribution, perform_apriori_analysis, get_items, get_list_items, get_items_input, update_csv

# Load data
data = pd.read_csv('shop.csv', parse_dates=['date'])
itemsList = get_items(data)
# Navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Channels", "Items", "New Entry"])

# Set default dates
min_date = data['date'].min().date()
max_date = data['date'].max().date()

# Date input
start_date = st.sidebar.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input('End Date', min_value=start_date, max_value=max_date, value=max_date)

# Filter data by date range
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
filtered_data = data[(data['date'].between(start_date, end_date))]

# Home page
if page == "Home":
    st.snow()
    st.title("Farahy Shop")
    search = st.text_input("Search", "")
    if search:
        filtered_data = filtered_data[filtered_data['items'].str.contains(search, case=False, na=False)]
    if filtered_data.empty:
        st.write("No results found.")
    else:
        n = 1
        if (len(filtered_data) > 1):
            n = st.slider('Number of rows to view', 1, len(filtered_data), len(filtered_data))
        edited_data = st.data_editor(filtered_data.head(n),key = 'data_editor')
        if st.button("Save Changes"):
            # Update original data
            data.update(edited_data)
            update_csv(data)

# Channels page
elif page == "Channels":
    channel = st.sidebar.selectbox('Select Channel', ['All'] + list(data['Channel'].unique()))    
    if channel != 'All':
        filtered_data = filtered_data[filtered_data['Channel'] == channel]

    if not filtered_data.empty:
        filtered_data.set_index('date', inplace=True)
        st.line_chart(filtered_data['Total Amount'].resample('M').sum())
        st.write(f'Summary statistics for {"All channels" if channel == "All" else channel}')
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

# Items page
elif page == "Items":
    st.title("Items Analysis")
    item = st.sidebar.selectbox('Select Item', pd.Series(filtered_data['items'].apply(parse_items).explode()).unique())
    if item:
        st.write(f'Distribution of {item}')
        plot_item_distribution(filtered_data, item)
# data entry page
elif page == "New Entry":
    st.title("New Entry")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        job_number = col1.number_input("Job Number", value=data['Job #'].max() + 1)
        date = col2.date_input("Date", value=pd.to_datetime('today'))
        cnls = data['Channel'].unique().tolist()
        channel = col3.selectbox("Channel", ["New Channel"] + cnls, index=None)
        if (channel == "New Channel"):
            channel = col3.text_input("Channel", "")
        with col4:
            ad = st.checkbox("Ad", value=False)
            rent = st.checkbox("Rent", value=False)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        amount = col1.number_input("Amount", value=0.0)
        shipping = col2.number_input("Shipping", value=0.0)
        paid_by = col3.number_input("Paid by Farahy", value=0.0)
        farahy_income = col4.number_input("Farahy Income", value=0.0)
    # Get items and amounts
    with st.container():
        items_str = get_items_input(itemsList)

    if st.button("Submit Entry"):
        new_entry = {
            "Job #": job_number,
            "date": pd.to_datetime(date),
            "Channel": channel,
            "Ad": (lambda x: 'Y' if x else 'N')(ad),
            "rent": (lambda x: 'Y' if x else 'N')(rent),
            "Amount": amount,
            "Shipping": shipping,
            "Total Amount": amount + shipping,
            "Paid by F": paid_by,
            "Farahy Income": farahy_income,
            "Item": items_str,
            "items" : get_list_items(items_str)
        }
        st.write(new_entry)
        # Append new entry to the DataFrame (Replace with your actual data appending logic)
        data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
        st.success("New entry added successfully!")
        st.balloons()
        st.write(data.tail(1))
        data.to_csv('shop.csv', index=False)
        # clear session state
        st.session_state.clear()
