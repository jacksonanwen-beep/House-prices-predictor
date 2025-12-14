from typing import Dict
from numpy import ndarray
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class Model:
    median_values: Dict[str,float]
    data_scaler: MinMaxScaler
    regression_model: LinearRegression

    def __init__(self):
        self.median_values = None
        self.data_scaler = MinMaxScaler()
        self.regression_model = LinearRegression()

    def train(self, data: DataFrame):
        input_data = get_input_columns_from_data(data)
        self.median_values = calculate_median_values(input_data)
        na_filled_training_data = self.fill_nas(input_data)
        self.fit_data_scaler_transform(na_filled_training_data)
        scaled_input_data = self.apply_data_scaling(na_filled_training_data)
        y_data = get_output_column_from_data(data)
        self.regression_model.fit(scaled_input_data,y_data)

    
    def predict(self, data: DataFrame) -> ndarray:
        input_columns = get_input_columns_from_data(data)
        nas_filled = self.fill_nas(input_columns)
        scaled_data = self.apply_data_scaling(nas_filled)
        return self.regression_model.predict(scaled_data)

    def fit_data_scaler_transform(self, data: DataFrame):
        self.data_scaler.fit_transform(data)

    def apply_data_scaling(self, data: DataFrame):
        return DataFrame(self.data_scaler.transform(data),columns=data.columns)

    def fill_nas(self, data: DataFrame) -> DataFrame:
        filled_df = {}
        for col in data.columns:
            median_of_column = self.median_values[str(col)]
            nas_filled = data[col].fillna(median_of_column)
            filled_df[str(col)] = nas_filled
        return DataFrame(filled_df)
                   
        # nas_filled = training_data[col_name].fillna(median_of_column)
        # training_data[col_name] = nas_filled



training_data, test_data = train_test_split(
    read_csv("./Data/train.csv"),
    test_size=0.2,
    random_state=1,
    shuffle=True
)

NUMBER_DATA_TYPES = ['int64','float64']


def get_datatype_for_column(column_name: str, dtypes) -> str:
    column_datatype = dtypes[column_name]
    return column_datatype.name


def get_input_columns_from_data(data: DataFrame) -> DataFrame:
    input_columns = []

    # collect only columns containing number datatypes into training_data_columns()
    for column_name in data.columns:
        column_dtype = get_datatype_for_column(column_name,data.dtypes)
        if column_dtype in NUMBER_DATA_TYPES:
            input_columns.append(str(column_name))

    # these columns aren't useful for predicting
    input_columns.remove("Id")
    input_columns.remove("MiscVal")

    # this is the column we're going to predict
    input_columns.remove("SalePrice")


    # remove columns which have too many Na's as the data is too sparse
    # for col_name in input_columns:
    #     column = data[col_name]
    #     if column.isna().sum() > 50:
    #         if col_name in input_columns:
    #             input_columns.remove(col_name)

    return data[input_columns]

def get_output_column_from_data(data: DataFrame) -> DataFrame:
    return data["SalePrice"]




def calculate_median_values(data: DataFrame) -> Dict[str,float]:
    median_values = {}
    # replace nas with median values
    for col_name in data.columns:
        median_values[str(col_name)] = training_data[col_name].dropna().median()
        
        # nas_filled = training_data[col_name].fillna(median_of_column)
        # training_data[col_name] = nas_filled
    return median_values



def plot_error_linear(predictions, actual, title):
    # calculate how far off we were for each guess
    error = predictions - actual

    # plot our errors
    # plt.close()
    plt.figure()
    error.plot(kind="kde", bw_method=0.05, title=title)
    plt.show()

# uncomment this to see visualisations of the distribution of our data# 
# for col_name in training_data.columns:
#     plt.close()
#     training_data[col_name].plot(kind="kde", bw_method=0.01, title=str(col_name))
#     plt.show()



model = Model()
model.train(training_data)
train_guesses = model.predict(training_data)
y_training_data = get_output_column_from_data(training_data)

plot_error_linear(train_guesses,y_training_data, "Train Data Errors")


test_guesses = model.predict(test_data)
y_test_data = get_output_column_from_data(test_data)

plot_error_linear(test_guesses,y_test_data, "Test Data Errors")

test_r2 = r2_score(y_true = y_test_data, y_pred = test_guesses)
test_mse = mean_squared_error(y_true = y_test_data, y_pred = test_guesses)
test_mae = mean_absolute_error(y_true = y_test_data, y_pred = test_guesses)

print(f"r2 = {test_r2}")
print(f"mean squared error = {test_mse}")
print(f"mean absolute error = {test_mae}")


input()