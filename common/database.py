#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Methods for accessing the database.

Project:        BVM Data Science Cup 2017
Description:
    This script contains some methods to easily access the database and retrieve
    information that is used by all tasks in the competition.
Author:         Christopher Harms
Email:          christopher.harms@skopos.de
Twitter:        @chrisharms

All rights reserved.

'''
import psycopg2 as pg
import pandas as pd
import numpy as np


def db_connection(hostname="dbserver", database="dsc2017"):
    """
    Connect to database and return an object for database connection
    :param hostname: Hostname or IP of PostgreSQL server
    :param database: Name of database
    :return: psycopg2-database connection
    """
    DB_USERNAME = "dbuser"
    DB_PASSWORD = "dbpassword"

    try:
        db = pg.connect(host=hostname, dbname=database, user=DB_USERNAME, password=DB_PASSWORD)
    except:
        print("Cannot connect to database.")
        return False
    
    return db


def get_patch_dates(db):
    """
    Get Patch dates from database
    :param db: object holding the connection to the database
    :return: array of patchdates
    """
    if not db:
        return False

    sql = "SELECT patch, patchversion, patchbuild, fromdate AS t FROM patches WHERE fromdate < '2015-12-24' "
    df = pd.io.sql.read_sql(sql, db)
    df['t'] = pd.to_datetime(df['t'])

    return df['t']


def get_item_sales(item_id, db, log_transform=False):
    """
    Get Item Sales for specific Item ID
    :param item_id: ID of item to get sales data for
    :param db: object holding the database connection
    :param log_transform: log-transform sales?
    :return: pandas dataframe holding the data
    """
    sql = "SELECT MIN(typeid) AS item, time AS ds, SUM(sales_units) AS y_raw, AVG(avgprice), MIN(minprice), MAX(maxprice), SUM(orders) FROM items WHERE typeid = {0} GROUP BY time ORDER BY time".format(
        item_id)
    df = pd.io.sql.read_sql(sql, db)
    if log_transform:
        df['y'] = np.log(df['y_raw'])
    else:
        df['y'] = df['y_raw']

    df['ds'] = pd.to_datetime(df['ds'])
    return df


def get_price_index(index, db, log_transform=False):
    """
    Get CPI values from database
    :param db: 
    :return: 
    """
    if not (index == "CPI" or index == "MPI" or index == "SPPI"):
        return False

    sql = f'''SELECT time AS ds, price AS y_raw FROM {index} ORDER BY time'''
    df = pd.io.sql.read_sql(sql, db)
    if log_transform:
        df['y'] = np.log(df['y_raw'])
    else:
        df['y'] = df['y_raw']

    df['ds'] = pd.to_datetime(df['ds'])
    df['item'] = index
    return df
