import pandas as pd
import psycopg2
from sqlalchemy import create_engine


def create_sqlalchemy_connection(conn_str_file):
    sqlalchemy_conn_str = open(conn_str_file,'r').read()
    sqlalchemy_conn = create_engine(sqlalchemy_conn_str)
    return sqlalchemy_conn

def query_best_parameters(conn_str_file, model_number=None):
    sqlalchemy_conn = create_sqlalchemy_connection(conn_str_file)
    if model_number != None:
        best_parameters = pd.read_sql('SELECT * FROM validation_metrics WHERE model_number = {} ORDER BY auc DESC LIMIT 1'.format(str(model_number)), 
                                  sqlalchemy_conn).to_dict(orient='records')[0]
    if model_number == None:
        best_parameters = pd.read_sql('SELECT * FROM validation_metrics ORDER BY auc DESC LIMIT 1'.format(str(model_number)), 
                                  sqlalchemy_conn).to_dict(orient='records')[0]
    del best_parameters['index']
    del best_parameters['auc']
    del best_parameters['epochs']
    del best_parameters['model_number']
    del best_parameters['free']
    return best_parameters