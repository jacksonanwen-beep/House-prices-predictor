from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


training_data = read_csv("./Data/train.csv")

NUMBER_DATA_TYPES = ['int64','float64']


def get_datatype_for_column(column_name: str, dtypes) -> str:
    column_datatype = dtypes[column_name]
    return column_datatype.name




training_data_columns = []


# collect only columns containing number datatypes into training_data_columns()
for column_name in training_data.columns:
    column_dtype = get_datatype_for_column(column_name,training_data.dtypes)
    if column_dtype in NUMBER_DATA_TYPES:
        training_data_columns.append(str(column_name))

# these columns aren't useful for predicting
training_data_columns.remove("Id")
training_data_columns.remove("MiscVal")

# this is the column we're going to predict
training_data_columns.remove("SalePrice")



# remove columns which have too many Na's in case they afffect the median for any reason
for col_name in training_data_columns:
    column = training_data[col_name]
    if column.isna().sum() > 50:
        if col_name in training_data_columns:
            training_data_columns.remove(col_name)


# replace nas with median values
for col_name in training_data_columns:
    median_of_column = training_data[col_name].dropna().median()
    nas_filled = training_data[col_name].fillna(median_of_column)
    training_data[col_name] = nas_filled



column_to_predict = training_data["SalePrice"]


# filter down the training data so we only use the columns useful for predicting 

training_data = training_data[training_data_columns]    

print(training_data)


# normalise our data
 
data_scaler = MinMaxScaler()
normalised_data = DataFrame(data_scaler.fit_transform(training_data),columns=training_data.columns)

print(normalised_data)

# uncomment this to see visualisations of the distribution of our data# for col_name in normalised_data.columns:
#     plt.close()
#     normalised_data[col_name].plot(kind="kde", bw_method=0.01, title=str(col_name))
#     plt.show()


# create and train out model against the training data
model = LinearRegression()
model.fit(normalised_data,column_to_predict)


# predict the sale prices for our training data
guesses = model.predict(normalised_data)


# calculate how far off we were for each guess
error = guesses - column_to_predict


# plot our errors
plt.close()
error.plot(kind="kde", bw_method=0.01, title=str(error))
plt.show()
print()
