
DROP_COLUMNS = [
    'id', 
    'date_account_created',
    'timestamp_first_active',
    'date_first_booking',
    'gender',
    'age',
    'signup_method',
    'signup_flow',
    'language', 
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'signup_app',
    'first_device_type', 
    'first_browser',
    'country_destination',
    'secs_elapsed'
]

LABEL_COLUMN = 'label_country_destination'

DATA_PATH = './data/'
METADATA_PATH = './metadata/metadata.pkl'

DEBUG = False