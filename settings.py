from class_imbalance import *

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

SAMPLING_METHOD = 'no_transform'
WEIGHTED = True

SAMPLING_MAPPING = {
    'no_transform': get_simple_data,
    'over_sample': get_over_sampled_data,
    'under_sample': get_under_sampled_data,
    'smote_sample': get_smote_data,
    'smotteenn_sample': get_smoteenn_data
}

RANDOM_SEARCH = False

LABEL_COLUMN = 'label_country_destination'
AGE_COLUMN = 'processed_age'

DATA_PATH = './data/'
METADATA_PATH = './metadata/metadata.pkl'

GRAPHS = 'graphs/'
STACK_PLOTS = 'graphs/stack_plots/'
METRICS = 'metrics/'
PROCESSED_DATA = 'processed_data/data.pkl'
CLASSIFICATION_REPORT = 'classification_report/'
IMBALANCE_CLASSIFICATION_REPORT = 'imbalance_classification_report/'

DEBUG = False
SAVE_METRICS = True