import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# Load your dataset
df = pd.read_csv("Bakery sales.csv")

# Convert the 'tanggal' column to datetime
df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M')
df = df.drop(['date','time'],axis=1)
df['date'] = df['Datetime'].dt.date

df['month'] = df['Datetime'].dt.month
df['month'] = df['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12),
                                          ('January','February','March','April','May','June','July','August',
                                          'September','October','November','December'))

df['weekday'] = df['Datetime'].dt.weekday
df['weekday'] = df['weekday'].replace((0,1,2,3,4,5,6),
                                          ('Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'))

df.drop('Datetime', axis = 1, inplace = True)

st.title("Market Basket Analysis with Apriori")

# Sidebar for user input
st.sidebar.header("Masukan Kriteria")
item = st.sidebar.selectbox("Pilih Item", df['article'].unique())
month = st.sidebar.selectbox("Bulan", df['month'].unique())
day = st.sidebar.selectbox("Hari", df['weekday'].unique())

# Function to get filtered data based on user input
def get_data(month, day, item):
    filtered = df.loc[(df["month"] == month) & (df["weekday"] == day) & (df["article"] == item)]
    return filtered

# Function to encode the data
def encode(x):
    return 1 if x > 0 else 0

# Function to parse list
def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

# Function to return item recommendations
def return_item_df(item_antecedents):
    data_filtered = rules[(rules["antecedents"].apply(lambda x: item_antecedents in x))]
    if not data_filtered.empty:
        antecedent = parse_list(data_filtered['antecedents'].values[0])
        consequent = parse_list(data_filtered['consequents'].values[0])
        return [antecedent, consequent]
    else:
        return ["No result", "No result"]

# Get user input
data = get_data(month, day, item)

if not data.empty:
    # Apply association analysis
    item_count = df.groupby(["ticket_number", "article"])["article"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='ticket_number', columns='article', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.astype("int32")
    item_count_pivot = item_count_pivot.applymap(encode)

    # Apriori Algorithm
    support = st.slider("Select Support Threshold", 0.0, 1.0, 0.2, 0.01)
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    # Association Rules
    metric = st.radio("Select Evaluation Metric", ["lift", "confidence", "support"])
    min_threshold = st.slider("Select Minimum Threshold", 0.0, 1.0, 0.5, 0.01)
    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

    # Display rules
    if not rules.empty:
        st.subheader("Generated Association Rules:")
        st.write(rules)
    else:
        st.warning("No rules generated for the selected criteria.")

    # Display recommendations
    st.subheader("Recommendations:")
    result = return_item_df(item)
    if result[0] != "No result":
        st.success(f"If a customer buys **{item}**, they may also buy **{result[0]}**.")
    else:
        st.warning("No recommendations found for the selected item.")
else:
    st.warning("No data available for the selected criteria. Please adjust your input.")
