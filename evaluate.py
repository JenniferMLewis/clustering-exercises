import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures


def get_rmse(y_train, target = "tax_value"):
    '''
    Takes in y_train and target (default: tax_value)
    '''
    y_train['baseline_mean'] = y_train[target].mean()
    y_train['baseline_median'] = y_train[target].median()

    # scores:
    rmse_mean = mean_squared_error(y_train.tax_value,
                                y_train['baseline_mean'], squared=False)
    rmse_med = mean_squared_error(y_train.tax_value,
                                y_train['baseline_median'], squared=False)

    print("RMSE Mean:")
    print(rmse_mean)
    print("----------------")
    print("RMSE Median:")
    print(rmse_med)
    print("----------------")
    if rmse_mean < rmse_med:
        print(f"RMSE Mean is lower so we'll use that : {rmse_mean}")
    elif rmse_med < rmse_mean:
        print(f"RMSE Median is lower so we'll use that : {rmse_med}")


def create_metric_df(y_train, y_validate, target = "tax_value"):
    '''
    Takes in a y_df (usually y_train), and a target (default: tax_value)
    calculates the baseline mean and baseline median, compares mean vs median,
    prints which value is lower (more accurate),
    then returns a metric_df comprised of the model name (baseline mean or baseline median), with RMSE score column and r^2 column.
    '''
    y_train['baseline_mean'] = y_train[target].mean()
    y_train['baseline_median'] = y_train[target].median()
    y_validate['baseline_mean'] = y_validate[target].mean()
    y_validate['baseline_median'] = y_validate[target].median()

    # scores:
    rmse_mean = mean_squared_error(y_train[target],
                                y_train['baseline_mean'], squared=False)
    rmse_med = mean_squared_error(y_train[target],
                                y_train['baseline_median'], squared=False)
    if rmse_mean < rmse_med:
        print(f"RMSE Mean is lower so we'll use that : {rmse_mean}")
        metric_df = pd.DataFrame(
        [
            {
                'model': 'baseline_mean',
                'rmse': mean_squared_error(y_train[target], y_train.baseline_mean),
                'r^2': explained_variance_score(y_train[target], y_train.baseline_mean)
            
            }
        ])
        return metric_df
    elif rmse_med < rmse_mean:
        print(f"RMSE Median is lower so we'll use that : {rmse_med}")
        metric_df = pd.DataFrame(
        [
            {
                'model': 'baseline_median',
                'rmse': mean_squared_error(y_train[target], y_train.baseline_median),
                'r^2': explained_variance_score(y_train[target], y_train.baseline_median)
            
            }
        ])
        return metric_df


def model_metrics(model,
                  model_name,
                  X_train, 
                  y_train, 
                  X_val, 
                  y_val, 
                  metric_df,
                  target = 'tax_value'):
    '''
    Takes in model, model_name (proper python naming format please), X,y train, X,y validate, 
    df to append [if you don't have a metric_df, run create_metric_df() first please.], and target (default: tax_value)
    Fits our training set, creates predictions using the SKLearn model given,
    adds the model scores to pre-existing metric_df,
    returns appended metric_df
    '''
    model.fit(X_train, y_train[target])
    in_sample_pred = model.predict(X_train)
    out_sample_pred = model.predict(X_val)
    y_train[model_name] = in_sample_pred
    y_val[model_name] = out_sample_pred
    print("y shape:")
    print(y_val.shape)
    print("----------")
    print("out of sample shape:")
    print(out_sample_pred.shape)
    rmse_val = mean_squared_error(
    y_val[target], out_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_val[target], out_sample_pred)
    return metric_df.append({
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }, ignore_index=True)


def polyfeatures(X_train, y_train, X_validate, y_validate, metric_df, model_name= "polynomial_regression_deg2", degree = 2):
    polyfeats = PolynomialFeatures(degree)
    X_train_quad = polyfeats.fit_transform(X_train)
    X_val_quad = polyfeats.transform(X_validate)
    metric_df = model_metrics(LinearRegression(), 
                  model_name,
                  X_train, 
                  y_train, 
                  X_validate, 
                  y_validate, 
                  metric_df)
    return metric_df

def model_metrics_all(X_train, y_train, X_val, y_val):
    metric_df = create_metric_df(y_train, y_val)
    metric_df = model_metrics(LinearRegression(normalize=True),
                  "linear_regression",
                  X_train, 
                  y_train, 
                  X_val, 
                  y_val, 
                  metric_df,
                  target = 'tax_value')
    metric_df = model_metrics(LassoLars(alpha=3.8),
                  "lasso_lars",
                  X_train, 
                  y_train, 
                  X_val, 
                  y_val, 
                  metric_df,
                  target = 'tax_value')
    metric_df = model_metrics(TweedieRegressor(power=1, alpha=0),
                  "glm",
                  X_train, 
                  y_train, 
                  X_val, 
                  y_val, 
                  metric_df,
                  target = 'tax_value')
    metric_df = polyfeatures(X_train, y_train, X_val, y_val, metric_df, model_name= "polynomial_regression_deg2", degree = 2)
    return metric_df

def rmse_in_out(X_train, y_train, X_validate, y_validate, target="tax_value"):

    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train[target])

    y_train['pred_lm'] = lm.predict(X_train)
    lm_rmse_train = mean_squared_error(y_train[target], y_train.pred_lm)**(1/2)

    y_validate['pred_lm'] = lm.predict(X_validate)
    lm_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", lm_rmse_train, 
        "\nValidation/Out-of-Sample: ", lm_rmse_validate)
    print("-----")

    lars = LassoLars(alpha=1.0)
    lars.fit(X_train, y_train[target])

    y_train['pred_lars'] = lars.predict(X_train)
    lars_rmse_train = mean_squared_error(y_train[target], y_train.pred_lars)**(1/2)

    y_validate['pred_lars'] = lars.predict(X_validate)
    lars_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", lars_rmse_train, 
        "\nValidation/Out-of-Sample: ", lars_rmse_validate)
    print("-----")

    glm = TweedieRegressor(power=1, alpha=0)

    glm.fit(X_train, y_train[target])

    y_train['pred_glm'] = glm.predict(X_train)
    glm_rmse_train = mean_squared_error(y_train[target], y_train.pred_glm)**(1/2)

    y_validate['pred_glm'] = glm.predict(X_validate)
    glm_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_glm)**(1/2)

    print("RMSE for GLM (Generalised Linar Model) using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", glm_rmse_train, 
        "\nValidation/Out-of-Sample: ", glm_rmse_validate)
    return y_train, y_validate


def plot_pred(y_validate, target = "tax_value"):
    plt.figure(figsize=(16,8))
    plt.plot(y_validate[target], y_validate.baseline_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_validate[target], y_validate[target], alpha=.5, color="black", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_validate[target], y_validate.pred_lm, 
                alpha=.5, color="red", s=10, label="Model: LinearRegression")
    plt.scatter(y_validate[target], y_validate.pred_lars, 
                alpha=.5, color="blue", s=10, label="Model: LassoLars")
    # plt.scatter(y_validate[target], y_validate.pred_glm, 
    #             alpha=.5, color="yellow", s=10, label="Model: TweedieRegressor")
    # plt.scatter(y_validate[target], y_validate.polynomial_regression_deg2, 
    #             alpha=.5, color="green", s=10, label="Model: 2nd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual Tax Value")
    plt.ylabel("Predicted Tax Value")
    plt.title("Where are predictions more extreme? More modest?")
    plt.show()


def plot_pred_hist(y_validate, target = "tax_value"):
    plt.figure(figsize=(16,8))
    plt.hist(y_validate[target], color='blue', alpha=.5, label="Actual Tax Value")
    plt.hist(y_validate.pred_lm, color='red', alpha=.5, label="Model: LinearRegression")
    # plt.hist(y_validate.pred_glm, color='yellow', alpha=.5, label="Model: TweedieRegressor")
    plt.hist(y_validate.pred_lars, color='green', alpha=.5, label="Model: LassoLars")
    plt.xlabel("Tax Value")
    plt.ylabel("Number of Properties")
    plt.title("Comparing the Distribution of Actual Tax Value to Distributions of Predicted Tax Value for the Top Models")
    plt.legend()
    plt.show()

def final_model(X_train, y_train, X_test, y_test, target = "tax_value"):
    lars = LassoLars(alpha=3.8)
    lars.fit(X_train, y_train[target])
    y_test['pred_lars'] = lars.predict(X_test)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test[target], y_test.pred_lars)**(1/2)

    print("RMSE for OLS Model using Lasso + Lars\nOut-of-Sample Performance: ", rmse_test)
