import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

from env import user, password, host


# ================   General   ==================

def get_url(db, user=user, password=password, host=host):
    '''
    Takes database name for input,
    returns url, using user, password, and host pulled from your .env file.
    PLEASE save it as a variable, and do NOT just print your credientials to your document.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def remove_outliers(df, col_list, k = 1.5):
    ''' 
    Accepts a Dataframe, a column list of columns you want to affect, 
    and a k variable (defines how far above and below the quartiles you want to go)[Default 1.5].
    Removes outliers from a list of columns in a dataframe 
    and return that dataframe.

    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range

        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def hist_plot(df, not_in=['fips', 'year_built', 'date']):
    '''
    Plots Histograms for columns in the input Data Frame, 
    all but categorical data [default: 'date', 'fips' and, 'year_built']
    '''
    plt.figure(figsize=(16, 3))

    cols = [col for col in df.columns if col not in not_in]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1 <-- Good to note
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        df[col].hist(bins=5)
        # We're looking for shape not actual details, so these two are set to 'off'
        plt.grid(False)
        plt.ticklabel_format(useOffset=False)
        # mitigate overlap: This is handy. Thank you.
        plt.tight_layout()

    plt.show()


def box_plot(df, cols = ['bedrooms', 'bathrooms', 'area', 'tax_value']
):
    ''' 
    Takes in a Data Frame, and list of columns [Default : bedrooms, bathrooms, area, and tax_value]
    Plots Boxplots of input columns.
    '''
    
    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        sns.boxplot(data=df[[col]])
        plt.grid(False)
        plt.tight_layout()

    plt.show()


# ================   Getting Data   =================

def get_mallcustomer_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''
    df = pd.read_sql('SELECT * FROM customers;', get_url('mall_customers'))
    return df.set_index('customer_id')

def get_zillow_data():
    '''
    Pulls all property details, along with legerror, transactiondate, aircon desc, 
    arch style, building class, heating sys desc, property land use, stories, and construction type.
    returns: a single Pandas DataFrame with the index set to the primary parcelid field
    '''
    filename="zillow.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = df.drop(columns="Unnamed: 0")
        return df
    else:
        df = pd.read_sql('''SELECT
        prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
    ) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                        AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL
    AND prop.longitude IS NOT NULL
    AND transactiondate <= '2017-12-31';''', get_url('zillow'))
        df.to_csv(filename)

        return df


def prepare_zillow(df):
    ''' 
    Prepares Zillow data for exploration
    Removes Outliers, Shows Distributions of Numeric Data via Histograms and Box plots,
    Converts bedrooms datatype from String to Float, Splits Data into Train, Validate, Test.
    Returns the Train, Validate, and Text Dataframes.
    '''

    # removing outliers
    df = remove_outliers(df, ['bedroomcnt', 'bathroomcnt', 'calculateds', 'tax_value'])
    
    hist_plot(df)
    box_plot(df)
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    return train, validate, test 


def wrangle_zillow():
    '''
    Acquire and prepare data from Zillow database for explore,
    Uses get_zillow_mvp and prepare_zillow_mvp functions.
    Returns Cleaned, Outlier Removed, Train, Validate, and Test Data Frames.
    '''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test


# ============ MISC =============

def df_split(train, validate, test, target="logerror"):
    '''
    Takes in train, validate, and test df, as well as target (default: "logerror")
    Splits them into X, y using target.
    Returns X, y of train, validate, and test.
    y sets returned as a proper DataFrame.
    '''
    X_train, y_train = train.drop(columns=target), train[target]
    X_validate, y_validate = validate.drop(columns=target), validate[target]
    X_test, y_test = test.drop(columns=target), test[target]
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def scale(df, columns_for_scaling = ['bedrooms', 'bathrooms', 'tax_value'], scaler = MinMaxScaler()):
    '''
    Takes in df, columns to be scaled (default: bedrooms, bathrooms, tax_value), 
    and scaler (default: MinMaxScaler(); others can be used ie: StandardScaler(), RobustScaler(), QuantileTransformer())
    returns a copy of the df, scaled.
    '''
    scaled_df = df.copy()
    scaled_df[columns_for_scaling] = scaler.fit_transform(df[columns_for_scaling])
    return scaled_df

def train_split(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test
