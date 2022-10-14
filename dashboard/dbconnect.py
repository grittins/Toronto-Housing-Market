
import os
from sqlalchemy import create_engine

def get_env_deets():
    if 'pwd' in os.environ:
        return{
            'NAME': os.environ['db_name'],
            'USER': os.environ['username'],
            'PASSWORD': os.environ['pwd'],
            'HOST': os.environ['host'],
            'PORT': os.environ['port'],
        }
# conda install SQLAlchemy
# row is parameter for raise_on_warnings to get warning feedback from database
def dbconnect():
    detail_dict=get_env_deets()
    postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username=detail_dict['USER'],password=detail_dict['PASSWORD'],ipaddress=detail_dict['HOST'],port=detail_dict['PORT'],dbname=detail_dict['NAME']))
    eng = create_engine(postgres_str)

    return (eng)


if __name__ == '__main__':
    dbconnect()