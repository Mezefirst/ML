#1 Setup - Import relevent modules
# The following code imports relevant modules that allow you to ran the colab
# If technical issuses are encountered running som of the code sections that follow and try running this section again 

import pandas as pd

# The following lines adjust the granularity of reporiting 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#2 Import the dataset

# The following code imports the dataset that is used in the colab.

training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Get statistics on the dataset.
# The following code returns basic statistics about the data in the dataframe.
training_df.describe()

#3 Task 1: Solution (run this code block to view) { display-mode: "form" }

print("""The following columns might contain outliers:

  * total_rooms
  * total_bedrooms
  * population
  * households
  * possibly, median_income

In all of those columns:

  * the standard deviation is almost as high as the mean
  * the delta between 75% and max is much higher than the
      delta between min and 25%.""")