import pandas as pd
import os
import env

def get_url(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

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

        return df.set_index('parcelid')


def get_info(df):
    print(f'''Peek at the DataFrame:
    {df.head()}
    ----
    Description:
    {df.describe()}
    ----
    Info:
    {df.info()}
    ----
    Shape:
    {df.shape}''')