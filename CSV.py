import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        if graph_type == 'scatter':
            st.write(sns.scatterplot(data=data))
        elif graph_type == 'bar':
            st.write(sns.barplot(data=data))
        elif graph_type == 'line':
            st.write(sns.lineplot(data=data))
        elif graph_type == 'pie':
            if np.issubdtype(data[column].dtype, np.number):
                st.write('Pie chart is not suitable for numerical data')
            else:
                st.write(data[column].value_counts().plot.pie(autopct="%.1f%%"))

        # Show the plot
        st.pyplot(plt)

if __name__ == "__main__":
    main()
