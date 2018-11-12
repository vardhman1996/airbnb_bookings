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
]


LIST_MAPPING = {
    'gender': ['MALE', 'FEMALE', 'OTHER'],
    'signup_method': ['basic'],
    'signup_flow': [2,3,6],
    'language': ['ca', 'en', 'fi', 'ja'],
    'affiliate_channel': ['direct', 'other', 'seo', 'sem-brand'],
    'affiliate_provider': ['direct', 'google', 'craigslist'],
    'first_affiliate_tracked': ['linked', 'untracked'],
    'signup_app': ['Web'],
    'first_device_type': ['Mac Desktop', 'Windows Desktop'],
    'first_browser': ['Chrome', 'Safari', 'Firefox']
}

STAT_COLS = [
    'gender',
    'signup_method',
    'signup_flow',
    'language',
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'signup_app',
    'first_device_type',
    'first_browser'
]

LABEL_MAPPING = {
     'US' : 0, 
     'FR' : 1, 
     'CA' : 2, 
     'GB' : 3, 
     'ES' : 4, 
     'IT' : 5, 
     'PT' : 6, 
     'NL' : 7,
     'DE' : 8, 
     'AU' : 9, 
     'NDF' : 10,
     'other' : 11 
}

CLASS_IMBALANCE = {
    0:0.7,
    1:1,
    2:1,
    3:1,
    4:1,
    5:1,
    6:1,
    7:1,
    8:1,
    9:1,
    10:0.5,
    11:1
}

LABEL_COLUMN = 'label_country_destination'

DATA_PATH = './data/'
METADATA_PATH = './metadata/metadata.pkl'

GRAPHS = 'graphs/'
STACK_PLOTS = 'graphs/stack_plots/'
METRICS = 'metrics/'


DEBUG = False