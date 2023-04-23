from datetime import datetime
import os
import sys
import pandas as pd
import numpy as np
import json
import boto3
from mask import mask_factory_build
from io import StringIO
from faker.factory import Factory

OUTPUT_BUCKET_NAME = 'my_bucket_name'
SEED = 125345
Faker = Factory.create
fake = Faker(['fr_FR'])
fake.seed(SEED)

#function that read the file parameters.json and return the parameters as a dictionary
def read_parameters() -> dict:
    with open(os.path.join(sys.path[0], 'parameters.json')) as f:
        parameters = json.load(f)
    return parameters


def generate_masks(table_parameters: dict, length: int) -> list[pd.Series]:
    masks_tree = mask_factory_build(table_parameters['masks'])
    probabilities = masks_tree.get_probability()
    cartesian_mask = np.random.choice(list(probabilities.keys()), length, p=list(probabilities.values()))
    masks_dict = masks_tree.set_distribution(cartesian_mask)
    return masks_dict

def init_df(table_parameters: dict) -> pd.DataFrame:
    length = table_parameters['length']
    #we setup a np array of the length provided in the parameters.json
    empty_series = np.zeros(length)
    #we init the dataframe with the np array and with a single column (named __index). Column should be removed before uploading the dataframe to the s3 bucket!
    df = pd.DataFrame(empty_series, columns=['__index'])
    df['__index'] = df.index
    return df


def compute_constant(df: pd.DataFrame, column_param: dict, column_name: str) -> pd.Series:
    if 'value' not in column_param:
        raise Exception("missing mandatory key: value for constant generation method of column : " + column_name)
    return pd.Series(column_param['value'], index=df.index)

def compute_id(df: pd.DataFrame, column_param: dict, column_name: str) -> pd.Series:
    if 'length' not in column_param:
        raise Exception("missing mandatory key: length for id generation method of column : " + column_name)
    #could use pystr_format() instead
    
    return df['__index'].apply("C0" + str(fake.pyint(1000, 9999)) if fake.pybool() else  str(fake.pyint(10000, 99999)))

def get_dependency(df: pd.DataFrame, dependency: str) -> pd.Series:
    table_name, column_name = dependency.split('.')
    if table_name.lower() == 'self':
        return df[column_name]
    else: 
        raise NotImplementedError("dependency not implemented")

def get_date_from_params(date_name:str, column_param: dict, dependencies: dict) -> datetime | pd.Series:
    def random_shift(dt: pd.Series):
    # Generate a random number days between 1 and 10 years
        days = np.random.randint(1, 360*10)
        # Shift the datetime by the random number of days
        return dt + pd.Timedelta(days=days)

    date = None
    if column_param[date_name] == 'dependency':
        dependency = get_dependency(dependencies['data'], column_param[date_name])
        date = dependency.apply(lambda row: random_shift(row), axis=1)

    if column_param[date_name] == 'current_date_minus_2_days':
        date = datetime.now() - datetime.timedelta(days=2)
    if column_param[date_name] == 'current_date':
        date = datetime.now()
    try:
        date = datetime.strptime(column_param[date_name], "%Y-%m-%d")
    except ValueError:
        raise Exception("max-date is not a valid date")
    return date

# This function deserve a way better implementation
def compute_ts(df: pd.DataFrame, column_param: dict, column_name: str) -> pd.Series:
    min_date = datetime.date(2010,1,1)
    max_date = datetime.now()
    if 'min-date' not in column_param:
        raise Exception("missing mandatory key: length for id generation method of column : " + column_name)
    if "dependency" in column_param:
        dependency_column = get_dependency(df, column_param["dependency"])
        min_date = get_date_from_params("min-date", column_param, {"data": dependency_column, "type":"after"})

    else:
        min_date = get_date_from_params("min-date", column_param)
        max_date = get_date_from_params("max-date", column_param)
        return df['__index'].apply(fake.date_between_dates(min_date, max_date))

def compute_column(df: pd.DataFrame, column_param: dict, column_name: str) -> pd.Series:
    if 'gen-method' not in column_param:
        raise Exception("missing mandatory key")
    if column_param['gen-method'] == 'constant':
        column = compute_constant(df, column_param['parameters'], column_name)
    if column_param['gen-method'] == 'id':
        column = compute_id(df, column_param['parameters'], column_name)
    if column_param['gen-method'] == 'random-timeserie':
        column = compute_ts(df, column_param['parameters'], column_name)
    return column

# Right now the  function is based on the dict order and we have to be sure that the order
# will not break dependency between columns. In order to make this more flexible and decouple order of columns and dependency we should use an Iterator pattern
# (see https://refactoring.guru/design-patterns/iterator for more information). 
# Be careful about priority conflict (i think breadth-first traversal is the best way to go)
def generate_columns(df: pd.DataFrame, table_parameters: dict, masks_dict: list[pd.Series]) -> pd.DataFrame:
    for column_name, column_param in table_parameters['fields'].items():
        if column_name in df.keys():
            raise Exception("column name already exists in the dataframe")
        df[column_name] = compute_column(df, column_param, column_name)
    return df

#function that generate a dataframe based on the parameters
def generate_df(table_parameters: dict) -> pd.DataFrame:
    df = init_df(table_parameters)
    masks_dict = generate_masks(table_parameters, len(df))
    df = generate_columns(df, table_parameters, masks_dict)
    
    return df.drop(columns=['__index'], inplace=True)

def upload_to_s3(df: pd.DataFrame, table_name: str) -> None:
    #upload the dataframe to s3 bucket
    csv_name = table_name + '.csv'
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(OUTPUT_BUCKET_NAME, csv_name).put(Body=csv_buffer.getvalue())

#main function that call the read_parameters function and generate a dataframe based on the parameters
def main() -> None:
    parameters = read_parameters()
    #generate a dataframe for each tables in parameters dictionary
    for table_name, table_parameters in parameters["tables"].items():

        df = generate_df(table_parameters)
        upload_to_s3(df, table_name)



if __name__ == '__main__':
    main()