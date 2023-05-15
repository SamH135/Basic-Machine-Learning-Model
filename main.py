import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(r"C:\Users\samue\PycharmProjects\melb_data.csv\melb_data.csv")
# print a summary of the data in Melbourne data
print("Describe the data: \n\n")
print(melbourne_data.describe())

print("Name of each column: \n\n")
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# By convention, the prediction target is called y
y = melbourne_data.Price

# The columns that are inputted into our model, and later used to make
# predictions, are called "features." In our case, those would be the
# columns used to determine the home price.
melbourne_features = ['Rooms', 'Bathroom', 'Landsize',
                      'Lattitude', 'Longtitude']

# By convention, this data is called X
X = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state
# to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("\n\nMaking predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print("\nMean Absolute Error:  " + str(mean_absolute_error(val_y, val_predictions)))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


best_mae = float('inf')  # Initialize with a high value
best_tree_size = None

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if my_mae < best_mae:
        best_mae = my_mae
        best_tree_size = max_leaf_nodes

print("\nBest tree size: " + str(best_tree_size) + "\n")

print("Fitting the final model with all data...\n")
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

print("Final model predictions are: \n")
print(final_model.predict(X.head()))
# get predicted prices on validation data
val_predictions = final_model.predict(val_X)
print("\nMean Absolute Error:  " + str(mean_absolute_error(val_y, val_predictions)))
