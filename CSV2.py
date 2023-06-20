import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def main():
    st.title('CSVデータ分析アプリ')

    # CSVデータのアップロード
    file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data)

        # 統計情報を表示する列を選択
        column = st.selectbox('分析する列を選択してください', data.columns)
        selected_data = data[column]

        if np.issubdtype(data[column].dtype, np.number):
            st.write('数値列の統計')
            st.write('最大値: ', np.max(selected_data))
            st.write('最小値: ', np.min(selected_data))
            st.write('平均: ', np.mean(selected_data))
            st.write('標準偏差: ', np.std(selected_data))

            # 変数が2つ以上ある場合の回帰分析
            if len(data.columns) > 1:
                input_features = st.multiselect('回帰分析の入力特徴を選択してください', data.columns, key='input_features')
                if len(input_features) > 10:
                    st.write('入力特徴は10個までにしてください')
                else:
                    output_feature = st.selectbox('回帰分析の出力特徴を選択してください', data.columns)
                    model = LinearRegression()
                    model.fit(data[input_features], data[output_feature])
                    st.write('特徴の回帰係数: ', model.coef_)

        else:
            st.write('非数値列の統計')
            st.write('総数: ', len(selected_data))
            st.write('ユニークな値の数: ', len(np.unique(selected_data)))

        # 可視化
        graph_type = st.selectbox('グラフの種類を選択してください', ['散布図', '棒グラフ', '折れ線グラフ', '円グラフ', '重回帰分析', '単回帰分析'])
        x_axes = st.multiselect('グラフのx軸を選択してください', data.columns, key='x_axes')
        y_axes = st.multiselect('グラフのy軸を選択してください', data.columns, key='y_axes')
        fig, ax = plt.subplots()  # 各プロットに新しい図を作成
        if graph_type == '散布図':
            for x in x_axes:
                for y in y_axes:
                    sns.scatterplot(x=data[x], y=data[y], ax=ax)
                    ax.set_title(f'{x}と{y}の散布図')
        elif graph_type == '棒グラフ':
            for x in x_axes:
                for y in y_axes:
                    sns.barplot(x=data[x], y=data[y],ax=ax)
                    ax.set_title(f'{x}と{y}の棒グラフ')
        elif graph_type == '折れ線グラフ':
            for x in x_axes:
                for y in y_axes:
                    sns.lineplot(x=data[x], y=data[y], ax=ax)
                    ax.set_title(f'{x}と{y}の折れ線グラフ')
        elif graph_type == '円グラフ':
            if np.issubdtype(data[column].dtype, np.number):
                st.write('円グラフは数値データには適していません')
            else:
                data[column].value_counts().plot.pie(autopct="%.1f%%", ax=ax)
                ax.set_title(f'{column}の円グラフ')
        elif graph_type == '重回帰分析':
            if len(input_features) > 1 and np.issubdtype(data[output_feature].dtype, np.number):
                for feature in input_features:
                    if np.issubdtype(data[feature].dtype, np.number):
                        sns.regplot(x=data[feature], y=model.predict(data[input_features]), ax=ax)
                        ax.set_title(f'{feature}と予測値の重回帰分析')
                    else:
                        st.write(f'重回帰分析は数値データに対してのみ適用可能です: {feature} 列は数値データではありません')
            else:
                st.write('重回帰分析には少なくとも2つの数値入力特徴が必要です')
        elif graph_type == '単回帰分析':
            if len(input_features) == 1 and np.issubdtype(data[output_feature].dtype, np.number):
                if np.issubdtype(data[input_features[0]].dtype, np.number):
                    sns.regplot(x=data[input_features[0]], y=data[output_feature], ax=ax)
                    ax.set_title(f'{input_features[0]}と{output_feature}の単回帰分析')
                else:
                    st.write(f'単回帰分析は数値データに対してのみ適用可能です: {input_features[0]} 列は数値データではありません')
            else:
                st.write('単回帰分析には1つの数値入力特徴が必要です')
        else:
            st.write('適切なグラフを作成できませんでした')

        # プロットを表示
        st.pyplot(fig)

if __name__ == "__main__":
    main()
