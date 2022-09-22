import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures



# =================== GENERAL ======================

alf = 0.05 # Rest well, sweet prince. [alf = alpha = α]

def col_range(df):
    '''
    Takes in a data frame, returns the 'describe' of the data frame with a new entry 'range'.
    'Range' is the difference between the 'max' and 'min' columns.
    '''
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    print(stats_df)

def create_new_columns(train, validate, test):
    # Bathrooms and Area
    train['area_by_bathroom'] = train['area']/train['bathrooms']
    validate['area_by_bathroom'] = validate['area']/validate['bathrooms']
    test['area_by_bathroom'] = test['area']/test['bathrooms']
    # Bedroom and Area
    train['area_by_beds'] = train['area']/train['bedrooms']
    validate['area_by_beds'] = validate['area']/validate['bedrooms']
    test['area_by_beds'] = test['area']/test['bedrooms']
    # Bedrooms and Bathrooms
    train["bed_and_bath"] = train['bedrooms'] + train['bathrooms']
    validate["bed_and_bath"] = validate['bedrooms'] + validate['bathrooms']
    test["bed_and_bath"] = test['bedrooms'] + test['bathrooms']
    # Area by Rooms 
    train["rooms_and_area"] = train['bed_and_bath']/train['area']
    validate["rooms_and_area"] = validate['bed_and_bath']/validate['area']
    test["rooms_and_area"] = test['bed_and_bath']/test['area']

    return train, validate, test

def correlation(train, sample, target="tax_value"):
    corr, p = stats.spearmanr(train[sample], train['tax_value'])
    print(f"Correlation: {round(corr, 2)}")
    print(f"P: {p}")

# =================== VIS EXPLORE ======================

def plot_variable_pairs(df, num = ['area', 'tax_value']):
    '''
    Takes in DF (Train Please,) and numerical column pair 
    [Preset Numerical Data is area, and tax_value, but a new list can be fed in.]
    Returns the plotted out variable pairs heatmap and numerical pairplot.
    '''
    df_corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(df_corr, cmap='Purples', annot = True, mask= np.triu(df_corr), linewidth=.5)
    plt.title("Correlations between variables")
    plt.show()
    
    sns.pairplot(df[num].sample(1_000), corner=True, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(df, cat = ["bedrooms", "bathrooms"], target = 'tax_value'):
    '''
    Takes a Data Frame,
    plots a continuous varible [default target value of tax_value](please enter an int/float column) as y
    sorted by categorical variable [default bedrooms, bathrooms] (Year_built not included, there's over 100 years, not exactly categorical to the scale we want) as x.
    Returns swarm plot, violin plot, and cat plot for each.
    '''
    for col in cat:
        sns.swarmplot(data=df.sample(800), x=col, y=target, s=3)
        plt.show()

        sns.violinplot(data=df.sample(1_000), x=col, y=target, s=3)
        plt.show()
        
        sns.catplot(data=df.sample(500), x=col, y=target, s=2)
        plt.show()

def explore_cat(df, cat = ['bedrooms', 'bathrooms'], target = 'tax_value'):
    '''
    Takes in dataframe (Remember, explore TRAIN!), 
    categorial columns (default is bedrooms, and bathrooms),
    target column (tax_value default), 
    returns printed value counts for each category in each column
    '''
    for col in cat:
        print(f"-=== {col} ===-")
        print(' ')
        print("Value Counts:")
        print("-------------")
        print(df[col].value_counts())
        print(" ")
        print("Percent of Total:")
        print("-----------------")
        print(df[col].value_counts(normalize=True)*100)
        print(" ")
        o = pd.crosstab(df[col], df[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        result = p < alf
        if result == True:
            print(' >> P is less than Alpha.')
        else:
            print(" >> P is greater than Alpha.")
        print(" ")
        sns.countplot(x=col, data=df)
        plt.title(col + ' counts')
        plt.show()
    
        sns.barplot(data=df, x=col, y=target)
        rate = df[target].mean()
        plt.axhline(rate, label= 'average ' + target + ' rate')
        plt.legend()
        plt.title(target + ' rate by ' + col)
        plt.show()

        print(' ')
        print('======================================')
        print(' ')
    print("-=== logerror ===-")
    print(' ')
    print("logerror Average:")
    print(df.logerror.mean())
    print("-------------")


def explore_num(df, num = ['area', 'tax_value']):
    '''
    Takes in DataFrame (Remember, explore w/ Train!), 
    numerical columns to explore (default area, tax_value).
    '''
    for col in num:
        sns.histplot(data=df, x=col, hue='bathrooms')
        plt.show()  

    cols = [col for col in num if col not in ['tax_value']]
    for col in cols:
        sns.scatterplot(data=df, x=col, y='tax_value', hue='bathrooms')
        plt.show()

def tabbing(X_train, y_train, x_target, y_target):
    return pd.crosstab(X_train[x_target], y_train[y_target], normalize='index').style.background_gradient(cmap='Greens')
    
def hypo_test_area(train):
    ''' 
    Takes in the train dataset 
    Returns T-Test results for test of value above and below the area median.
    '''
    
    area_above_avg = train[train.area > train.area.median()].tax_value
    area_below_avg = train[train.area <= train.area.median()].tax_value

    # Check for equal variances
    s, pval = stats.levene(area_above_avg, area_below_avg)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(area_above_avg, area_below_avg, equal_var=(pval >= alf))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alf and t > 0:
        print('''We can Reject the Null Hypothesis.
This suggests there is more value in homes with above the median area than homes with median and below area.''')
    else:
        print('''Failed to reject the Null Hypothesis.
This suggests there is more value in homes with median and below area than homes with above the median area.''')

def hypo_test_bathrooms(train):
    ''' 
    Takes in the train dataset.
    Returns T-Test comparing value above the median number of bathrooms, and below it.
    '''
    # Create the samples
    bath_above = train[train.bathrooms > train.bathrooms.median()].tax_value
    bath_below = train[train.bathrooms <= train.bathrooms.median()].tax_value

    # Check for equal variances
    s, pval = stats.levene(bath_above, bath_below)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(bath_above, bath_below, equal_var=(pval >= alf))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alf and t > 0:
        print('''Reject the Null Hypothesis.
This suggests there is more value in homes with above median number of bathrooms than homes with below the median number bathrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
This suggests there is more value in homes with below median number of bathrooms than homes with above the median number of bathrooms.''')

def hypo_test_bedrooms(train):
    ''' 
    Takes in the train dataset.
    Returns T-Test results of value above the median number of bedrooms and below the median number.
    '''
    bed_above = train[train.bedrooms > train.bedrooms.median()].tax_value
    bed_below = train[train.bedrooms <= train.bedrooms.median()].tax_value

    s, pval = stats.levene(bed_above, bed_below)

    t, p = stats.ttest_ind(bed_above, bed_below, equal_var=(pval >= alf))
    if p/2 < alf and t > 0:
        print('''Reject the Null Hypothesis.
This suggests there is more value in homes with above median number of bedrooms than homes with below median bedrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
This suggests there is more value in homes with below median number of bedrooms than homes with above median bedrooms.''')

def hypo_test_area_bath_compare(train):
    '''
    Takes in Train dataframe.
    Returns comparision between Value of houses above median area and median number of bathrooms, 
    as well as Value of houses below median area and median number of bathrooms.
    '''
    # area vs bathroom average value.
    area_above_avg = train[train.area > train.area.median()].tax_value
    area_below_avg = train[train.area <= train.area.median()].tax_value
    bath_above = train[train.bathrooms > train.bathrooms.median()].tax_value
    bath_below = train[train.bathrooms <= train.bathrooms.median()].tax_value

    # Checking equal variance for both above and below .
    s, pval_a = stats.levene(area_above_avg, bath_above)
    s, pval_b = stats.levene(area_below_avg, bath_below)

    # Running T-Tests for both above and below
    t_a, p_a = stats.ttest_ind(area_above_avg, bath_above, equal_var=(pval_a >= alf))
    t_b, p_b = stats.ttest_ind(area_below_avg, bath_below, equal_var=(pval_b >= alf))

    # Evaluating results for above, then below.
    if p_a/2 < alf and t_a > 0:
        print('''Reject the Null Hypothesis.
This suggests there is more value in homes with above median area than homes with above median bathrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
This suggests there is more value in homes with above median bathrooms than homes with above median area.''')

    if p_b/2 < alf and t_b > 0:
        print('''
Reject the Null Hypothesis.
This suggests there is more value in homes with below median area than homes with below median bathrooms.''')
    else:
        print('''
Fail to reject the Null Hypothesis.
This suggets there is more value in homes with below median bathrooms than homes with below median area.''')

def hypo_test_area_bed_compare(train):
    '''
    Takes in Train dataframe.
    Returns comparision between Value of houses above median area and median number of bedrooms, 
    as well as Value of houses below median area and median number of bedrooms.
    '''
    # area vs bedroom average value.
    area_above_avg = train[train.area > train.area.median()].tax_value
    area_below_avg = train[train.area <= train.area.median()].tax_value
    bed_above = train[train.bedrooms > train.bedrooms.median()].tax_value
    bed_below = train[train.bedrooms <= train.bedrooms.median()].tax_value

    # Checking equal variance for both above and below .
    s, pval_a = stats.levene(area_above_avg, bed_above)
    s, pval_b = stats.levene(area_below_avg, bed_below)

    # Running T-Tests for both above and below
    t_a, p_a = stats.ttest_ind(area_above_avg, bed_above, equal_var=(pval_a >= alf))
    t_b, p_b = stats.ttest_ind(area_below_avg, bed_below, equal_var=(pval_b >= alf))

    # Evaluating results for above, then below.
    if p_a/2 < alf and t_a > 0:
        print('''Reject the Null Hypothesis.
This suggests there is more value in homes with above median area than homes with above median bedrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
This suggests there is more value in homes with above median bedrooms than homes with above median area.''')

    if p_b/2 < alf and t_b > 0:
        print('''
Reject the Null Hypothesis.
This suggests there is more value in homes with below median area than homes with below median bedrooms.''')
    else:
        print('''
Fail to reject the Null Hypothesis.
This suggets there is more value in homes with below median bedrooms than homes with below median area.''')

def hypo_test_bed_bath_compare(train):
    '''
    Takes in Train dataframe.
    Returns comparision between Value of houses above median area and median number of bathrooms, 
    as well as Value of houses below median area and median number of bathrooms.
    '''
    # bedroom vs bathroom average value.
    bed_above_avg = train[train.bedrooms > train.bedrooms.median()].tax_value
    bed_below_avg = train[train.bedrooms <= train.bedrooms.median()].tax_value
    bath_above = train[train.bathrooms > train.bathrooms.median()].tax_value
    bath_below = train[train.bathrooms <= train.bathrooms.median()].tax_value

    # Checking equal variance for both above and below .
    s, pval_a = stats.levene(bed_above_avg, bath_above)
    s, pval_b = stats.levene(bed_below_avg, bath_below)

    # Running T-Tests for both above and below
    t_a, p_a = stats.ttest_ind(bed_above_avg, bath_above, equal_var=(pval_a >= alf))
    t_b, p_b = stats.ttest_ind(bed_below_avg, bath_below, equal_var=(pval_b >= alf))

    # Evaluating results for above, then below.
    if p_a/2 < alf and t_a > 0:
        print('''Reject the Null Hypothesis.
This suggests there is more value in homes with above median bedrooms than homes with above median bathrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
This suggests there is more value in homes with above median bathrooms than homes with above median bedrooms.''')

    if p_b/2 < alf and t_b > 0:
        print('''
Reject the Null Hypothesis.
This suggests there is more value in homes with below median bedrooms than homes with below median bathrooms.''')
    else:
        print('''
Fail to reject the Null Hypothesis.
This suggets there is more value in homes with below median bathrooms than homes with below median bedrooms.''')

# ============== Feature Selection ====================
# Do not rely solely on these.

def select_kbest(X_train, y_train, k_features):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using SelectKBest from sklearn. 
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    kbest.fit(X_train, y_train)
    
    print(X_train.columns[kbest.get_support()].tolist())
    
    
def select_rfe(X_train, y_train, k_features, model = LinearRegression()):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using Recursive Feature Elimination from sklearn. 
    '''
    
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X_train, y_train)
    
    print(X_train.columns[rfe.support_].tolist())
    
    
def select_sfs(X_train, y_train, k_features, model = LinearRegression()):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using Sequential Feature Selector from sklearn. 
    '''
    sfs = SequentialFeatureSelector(model, n_features_to_select=k_features)
    sfs.fit(X_train, y_train)
    
    print(X_train.columns[sfs.support_].tolist())

def select_best(X_train, y_train, k_features, model = LinearRegression()):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Optional Model selection (default: LinearRegression())
    Runs select_kbest, select_rfe, and select sfs functions to find the different bests.
    Returns bests.
    '''
    print("KBest:")
    print(select_kbest(X_train, y_train, k_features))
    print(" ")
    print("RFE:")
    print(select_rfe(X_train, y_train, k_features, model))
    print(" ")
    print("SFS:")
    print(select_sfs(X_train, y_train, k_features, model))
