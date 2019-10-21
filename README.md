# Automated Blockchain Transactions with Parallel model comparison for best price Forecast of Bitcoin

In this project, I attempt to apply machine-learning algorithms to predict Bitcoin price and thus
automating the flow of transaction (like buying, selling or long, short) of Bitcoins based on the
predicted deviations is the forecasted price. For the first phase of the investigation, I aimed to
understand and better identify daily trends in the Bitcoin market while gaining insight into the
most optimal methods and techniques to successfully predict Bitcoin price. At the same time, a
comparison has also been drawn to gauge the performance of the most popular machine learning
algorithms which can be used to predict the Bitcoin prices. Three prominent models were tested
for this purpose: Bagging Algorithm (Random Forest), Boosting Algorithm (XGBoost) and Deep
Learning Algorithm (RNN with LSTM). The best results are observed in the case of Deep
Learning algorithm with a maximum accuracy of 99.23%, which is significantly higher than
most of the prevalent models today. In the second phase, we developed an automated signal
mechanism which is capable of producing a signal whether to short or long Bitcoins on any
given day. The signals have been classified into three categories: Weak, Moderate and Strong.

#### Please refer to the report in this repository for detailed analysis.

#### Data Link:
https://www.kaggle.com/mczielinski/bitcoin-historical-data/data

#### INTRUCTIONS TO RUN THE CODE:
To run the code completely, pleae make sure that you have all of the following libraries installed on your system:
- numpy
- pandas
- statsmodels.api
- scipy
- sklearn
- keras
- tensorflow
- matplotlib
- plotly
- xgboost

You can find the dataset used for this project at: https://www.kaggle.com/mczielinski/bitcoin-historical-data/data
You can also find the same dataset (directly in CSV format) used for this project at this drive link: https://drive.google.com/file/d/1XBdBciOcw7vWA1jyIwyRM4ZbHiyG_JfO/view?usp=sharing

Moreover, sufficient number of comments have been added at each step for the user so that the code is clear at every step.

Prefer using Python version > 3.0.0

Images for all the visualizations have been attached in the repository and all the results and comparisons have been documented in the report.
