import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression

def main():
    st.title('CSV Data Analysis App')

    # Upload CSV data
    file = st.file_uploader("Upload a CSV file", type=['csv'])
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data)

        # Select column to view statistics
        column = st.selectbox('Select column for analysis', data.columns)
        selected_data = data[column]

        if np.issubdtype(data[column].dtype, np.number):
            st.write('Statistics for numerical column')
            st.write('Max: ', np.max(selected_data))
            st.write('Min: ', np.min(selected_data))
            st.write('Mean: ', np.mean(selected_data))
            st.write('Std deviation: ', np.std(selected_data))

            # Regression analysis for two or more variables
            if len(data.columns) > 1:
                input_feature = st.selectbox('Select input feature for regression', data.columns)
                output_feature = st.selectbox('Select output feature for regression', data.columns)
                model = LinearRegression()
                model.fit(data[[input_feature]], data[output_feature])
                st.write('Regression coefficient for features: ', model.coef_)

        else:
            st.write('Statistics for non-numerical column')
            st.write('Total count: ', len(selected_data))
            st.write('Unique count: ', len(np.unique(selected_data)))

        # Visualization
        graph_type = st.selectbox('Select graph type', ['scatter', 'bar', 'line', 'pie'])
        x_axes = st.multiselect('Select x-axes for the graph', data.columns, key='x_axes')
        y_axis = st.selectbox('Select y-axis for the graph', data.columns, key='y_axis')
        fig, ax = plt.subplots()  # Create a new figure for each plot
        if graph_type == 'scatter':
            for x in x_axes:
                sns.scatterplot(x=data[x], y=data[y_axis], ax=ax)
        elif graph_type == 'bar':
            for x in x_axes:
                sns.barplot(x=data[x], y=data[y_axis], ax=ax)
        elif graph_type == 'line':
            for x in x_axes:
                sns.lineplot(x=data[x], y=data[y_axis], ax=ax)
        elif graph_type == 'pie':
            if np.issubdtype(data[column].dtype, np.number):
                st.write('Pie chart is not suitable for numerical data')
            else:
                data[column].value_counts().plot.pie(autopct="%.1f%%", ax=ax)
        else:
            st.write('Appropriate graph could not be created')

        # Show the plot
        st.pyplot(fig)

if __name__ == "__main__":
    main()
