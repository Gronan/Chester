# 
# This file is part of the Gronan/Chester distribution (https://github.com/Gronan/Chester).
# Copyright (c) 2023 Ronan TREILLET.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import numpy as np
import json
import boto3
from mask import mask_factory_build, MissingMaskLengthPropertyException, MaskGroup, MissingMaskTypePropertyException, \
    MissingMaskChildrenPropertyException, mask_gen_factory
from io import StringIO
from faker import Faker

OUTPUT_BUCKET_NAME = 'rt-bmw-chester-data'
SEED = 125345
fake = Faker(['fr_FR'])
# Set the seed value of the shared `random.Random` object
# across all internal generators that will ever be created
Faker.seed(SEED)


class PropertyException(Exception):
    def __init__(self):
        super().__init__('MaskGroupShouldHaveChildrenException')


# function that read the file parameters.json and return the parameters as a dictionary
def read_parameters() -> dict:
    with open(os.path.join(sys.path[0], 'parameters.json')) as f:
        parameters = json.load(f)
    return parameters


def generate_masks(table_parameters: dict, length: int) -> dict:
    masks_tree = mask_factory_build(table_parameters['masks'])
    try:
        assert 'gen_method' in table_parameters['gen_method']
    except AssertionError as e:
        raise MissingMaskChildrenPropertyException()
    cartesian_mask = mask_gen_factory(masks_tree, length, table_parameters['gen_method'])
    masks_dict = masks_tree.set_distribution(cartesian_mask)
    return masks_dict


def sum_mask_length(mask_list: list) -> int:
    length = 0
    try:
        assert 'children' in mask_list
    except AssertionError as e:
        raise MissingMaskChildrenPropertyException()
    for mask in mask_list['children']:
        try:
            try:
                assert 'type' in mask
            except AssertionError as e:
                raise MissingMaskTypePropertyException()
            if mask['type'] in MaskGroup.types:
                length = sum_mask_length(mask['children'])
            else:
                assert 'length' in mask
                length += mask['length']
        except AssertionError as e:
            raise MissingMaskLengthPropertyException()
    return length


def init_length_df(table_parameters: dict) -> int:
    length = 0
    if isinstance(table_parameters['length'], int):
        length = table_parameters['length']
    elif isinstance(table_parameters['length'], str):
        if table_parameters['length'] == "masks_addition" and table_parameters['masks'] is not None:
            length = sum_mask_length(table_parameters['masks'])
    return length


def init_df(table_parameters: dict) -> pd.DataFrame:
    length = init_length_df(table_parameters)
    # we set up a np array of the length provided in the parameters.json
    empty_series = np.zeros(length)
    # we initiate the dataframe with the np array and with a single column (named __index). Column should be removed
    # before uploading the dataframe to the s3 bucket!
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
    length = column_param['length']
    if not isinstance(length, int):
        try:
            length = int(length)
        except Exception:
            raise Exception("length must be an integer")

    # could use pystr_format() instead
    def gen_id(row) -> str:
        return "C0" + str(fake.pyint(1, 10 ** (length - 2) - 1)).rjust(length - 2, "0") if fake.pybool() else str(
            fake.pyint(1, 10 ** (length) - 1))

    return df['__index'].apply(gen_id)


def get_dependency(df: pd.DataFrame, dependency: str) -> pd.Series:
    table_name, column_name = dependency.split('.')
    if table_name.lower() == 'self':
        return df[column_name]
    else:
        raise NotImplementedError("dependency not implemented")


def gen_ts_from_series(dependency: pd.Series,
                       is_positive: bool = True,
                       range: int = 360 * 10,
                       max_date: datetime | pd.Series = None,
                       min_date: datetime | pd.Series = None
                       ) -> pd.Series:
    # we create a closure to be used as a lambda function
    def random_shift(row: pd.Series):
        # Generate a random number days between 1 and 10 years (by default)
        days = np.random.randint(1, range)
        # Shift the datetime by the random number of days
        if is_positive:
            added_time = row + pd.Timedelta(days=days)
        else:
            added_time = row - pd.Timedelta(days=days)
        return added_time

    if is_positive:
        assert max_date is not None, "max_date is not defined"
    # create a timeserie with random variation (positive or negative) based on the dependency 
    date = dependency.apply(random_shift)
    # if there is a max_date make sure we don't go beyond it
    if max_date is not None:
        if isinstance(max_date, datetime):
            max_date = pd.Series(max_date, index=dependency.index)
        date = np.where(date > max_date, max_date, date)
    # same for min_date
    if min_date is not None:
        if isinstance(min_date, datetime):
            min_date = pd.Series(min_date, index=dependency.index)
        date = np.where(date < min_date, min_date, date)
    return date


def get_date_from_params(date_name: str, column_param: dict) -> datetime | pd.Series:
    date = None
    if column_param[date_name] == 'dependency':
        return date
    if column_param[date_name] == 'current_date_minus_2_days':
        date = datetime.now() - timedelta(days=2)
        return date
    if column_param[date_name] == 'current_date':
        date = datetime.now()
        return date
    else:
        try:
            date = datetime.strptime(column_param[date_name], "%Y-%m-%d")
        except ValueError:
            raise Exception("max-date is not a valid date")
    return date


# This function deserve a way better implementation
def compute_ts(df: pd.DataFrame, column_param: dict, column_name: str) -> pd.Series:
    min_date = datetime(2010, 1, 1)
    max_date = datetime.now()
    if 'min-date' not in column_param:
        raise Exception(
            "missing mandatory key: min-date for random-timeserie generation method of column : " + column_name)
    if "dependency" in column_param:
        dependency_column = get_dependency(df, column_param["dependency"])
        max_date = get_date_from_params("max-date", column_param)
        date = gen_ts_from_series(dependency_column, is_positive=True, max_date=max_date)
        return date
    else:
        min_date = get_date_from_params("min-date", column_param)
        max_date = get_date_from_params("max-date", column_param)

        # we create a closure to fix max_date and min_date for this column
        def gen_fake_date(row) -> datetime:
            return fake.date_time_between(min_date, max_date)

        date = df['__index'].apply(gen_fake_date)
        return date


def compute_hash(df: pd.DataFrame) -> pd.Series:
    def gen_hash(row) -> str:
        return fake.sha256()

    column = df['__index'].apply(gen_hash)
    return column


def extract_random_list_params(params):
    weights = None
    values = None
    if isinstance(params, list):
        values = params
    if isinstance(params, dict):
        weights = list(params.values())
        values = list(params.keys())
    if isinstance(params, str):
        values = [params]
    if isinstance(params, int):
        values = [str(params)]
    return values, weights


def get_masked_random_list(df, column_param: dict, column_name: str, masks_dict: list[pd.Series]):
    if 'masks' not in column_param:
        raise Exception(
            "missing mandatory key: masks for masked_random_list generation method of column : " + column_name)
    if len(masks_dict) == 0:
        raise Exception("masks_dict is empty")

    column = None

    else_params = None
    else_mask = []
    has_else = False
    for mask_name, mask_params in column_param['masks'].items():
        # init column with the first mask avalaible
        if column is None and mask_name != "else":
            column = np.empty(masks_dict[mask_name].shape)
            column[:] = np.nan

        if mask_name == "else":
            has_else = True
            else_params = mask_params
            continue
        else:
            # get the mask from the masks_dict
            if mask_name not in masks_dict:
                raise Exception("mask not found in masks_dict")
            np_mask = masks_dict[mask_name]
            # append the mask to the list because if there is an else mask we need to compute it based on the others
            else_mask.append(np_mask)
            values, weights = extract_random_list_params(mask_params['values'])
            # suboptimal because we use a distribution of the size of the length in parameters.json when we could just create a distribution of the size of the mask
            column = np.where(np_mask, np.random.choice(values, column.size, p=weights), column)

    if has_else and len(else_mask) > 0:
        # compute else mask based on others
        values, weights = extract_random_list_params(else_params['values'])
        # get the opposite mask of the others masks
        rest_mask = ~np.any(else_mask, axis=0)
        # suboptimal because we use a distribution of the size of the length in parameters.json when we could just create a distribution of the size of the mask
        column = np.where(rest_mask, np.random.choice(values, column.size, p=weights), column)
    return column


def get_random_list(df, column_param: dict, column_name: str):
    column = None
    if 'values' not in column_param:
        raise Exception("missing mandatory key: values for random_list generation method of column : " + column_name)
    values, weights = extract_random_list_params(column_param['values'])
    column = np.random.choice(values, df.shape[0], p=weights)
    return column


def compute_column(df: pd.DataFrame, column_param: dict, column_name: str, masks_dict: list[pd.Series]) -> pd.Series:
    column = None
    try:
        if 'gen-method' not in column_param:
            raise Exception("missing mandatory key")
        if column_param['gen_method'] == 'constant':
            column = compute_constant(df, column_param['parameters'], column_name)
        if column_param['gen_method'] == 'id':
            column = compute_id(df, column_param['parameters'], column_name)
        if column_param['gen_method'] == 'random-timeserie':
            column = compute_ts(df, column_param['parameters'], column_name)
        if column_param['gen_method'] == 'hash':
            column = compute_hash(df)
        if column_param['gen_method'] == 'masked_random_list':
            column = get_masked_random_list(df, column_param['parameters'], column_name, masks_dict)
        if column_param['gen_method'] == 'random_list':
            column = get_random_list(df, column_param['parameters'], column_name)
        return column
    except UnboundLocalError as e:
        raise NotImplementedError(
            "missing implementation for column : " + column_name + " with type :" + column_param['gen-method'])

    # Right now the  function is based on the dict order and we have to be sure that the order


# will not break dependency between columns. In order to make this more flexible and decouple order of columns and dependency we should use an Iterator pattern
# (see https://refactoring.guru/design-patterns/iterator for more information). 
# Be careful about priority conflict (i think breadth-first traversal is the best way to go)
def generate_columns(df: pd.DataFrame, table_parameters: dict, masks_dict: list[pd.Series] | dict) -> pd.DataFrame:
    for column_name, column_param in table_parameters['fields'].items():
        if column_name in df.keys():
            raise Exception("column name already exists in the dataframe")
        df[column_name] = compute_column(df, column_param, column_name, masks_dict)
    return df


# function that generate a dataframe based on the parameters
def generate_df(table_parameters: dict) -> (pd.DataFrame, dict):
    df = init_df(table_parameters)
    masks_dict = generate_masks(table_parameters, len(df))
    df = generate_columns(df, table_parameters, masks_dict)
    df.drop(columns=['__index'], inplace=True)
    return df, masks_dict


def upload_to_s3(df: pd.DataFrame, table_name: str) -> None:
    # upload the dataframe to s3 bucket
    csv_name = table_name + '.csv'
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(OUTPUT_BUCKET_NAME, csv_name).put(Body=csv_buffer.getvalue())


# main function that call the read_parameters function and generate a dataframe based on the parameters
def main() -> None:
    parameters = read_parameters()
    masks = {}

    # generate a dataframe for each tables in parameters dictionary
    for table_name, table_parameters in parameters["tables"].items():
        df, table_mask = generate_df(table_parameters)
        masks[table_name] = table_mask
        upload_to_s3(df, table_name)


if __name__ == '__main__':
    main()
