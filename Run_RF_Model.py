
# Random Forest with Variables to see if I can predict reserovir operations sooner
# JCS
# Sep 25, 2023

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# sci py with multivariate linear regression
# XGBoosting tree --> gradient boosting regressor (do one by one and see if that works better or worse)
# bring scatter plots (RF for all on own and XGboost for all on one)
# packages
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import joblib
import seaborn as sn


from sklearn.metrics import accuracy_score
import math

# CHANGE VARIABLE NAMES TO BE p1, p2, p3

def create_sinusodial_curve(  alpha, beta,  mu, bound_variable):
    final_df = pd.DataFrame()
    final_df['epiweek'] = range(1,53)
    final_df[bound_variable] = np.nan
    counter = 0
    for i in range(1, 53):
        storage = alpha + beta * np.sin(2* math.pi *i/52) + mu* np.cos(2* math.pi *i/52) 
        final_df.iloc[counter, 1] = storage
        counter = counter +1
    return (final_df)

def create_sinusodial_curve_constrained(  alpha, beta,  mu, max, min,bound_variable):
    final_df = pd.DataFrame()
    final_df['epiweek'] = range(1,53)
    final_df[bound_variable] = np.nan
    counter = 0
    for i in range(1, 53):
        storage = alpha + beta * np.sin(2* math.pi *i/52) + mu* np.cos(2* math.pi *i/52) 
        if storage > max:
            storage = max
        if storage < min:
            storage = min
        final_df.iloc[counter, 1] = storage
        counter = counter +1
    return (final_df)

# rf_df = pd.read_csv('Data/RF/random_forest_inputs_all.csv')
# turner_params = pd.read_csv('Data/RF/params_glolakes_no_inf.csv')


# FOR ICESAT
rf_df = pd.read_csv('Data/RF/random_forest_inputs_geodar_all.csv') # all geodar

turner_params = pd.read_csv('Data/RF/params_glolakes_no_inf_iceSAT.csv')

turner_params = pd.read_csv('Data/RF/Turner_ResOpsUS_params.csv')
turner_params['geodar_id'] = np.nan
geodar_hydrolakes = 'Data/RF/geodar_hydrolakes.csv'
linkage = pd.read_csv(geodar_hydrolakes)
geodar_ids = linkage['id_v11']
for index in range(0, len(turner_params)):
    hydrolake_id = turner_params.iloc[index,3 ]
    grand_id = turner_params.iloc[index,0]
    if math.isnan(grand_id) == True:
        continue
    else:
        linkage_index = linkage[linkage['id_grd_v13'] == grand_id]
        geodar_id = linkage_index.iloc[0,1]
        turner_params.iloc[index,11 ] = geodar_id
rf_df = pd.merge(turner_params,rf_df, on = 'geodar_id')

# drop_columns


rf_df = rf_df.drop(columns=['geodar_id', 'Unnamed: 0', 'grand_id_x', 'grand_id_y'])

### RUN RF #### 
rf_df = pd.concat([turner_params.iloc[:,1:15],rf_df.iloc[:, 9:49]], axis = 1)
rf_df = rf_df.dropna(subset=['cap', 'area']) # removed na in cap and area
rf_df['dom_autumn'] = rf_df['dom_autumn'].fillna(0)# made domestic demand 0 if its missing
rf_df['dom_winter'] = rf_df['dom_winter'].fillna(0) # made domestic demand 0 if its missing
rf_df['dom_sumnmer'] = rf_df['dom_sumnmer'].fillna(0) # made domestic demand 0 if its missing
rf_df['dom_spring'] = rf_df['dom_spring'].fillna(0)# made domestic demand 0 if its missing
rf_df['autumn_inflow']= rf_df['autumn_inflow'].fillna(0) # made domestic demand 0 if its missing

#Daans comment: making demand 0 when its missing might not be the way to work with missing data. Especially since the missing data mechanism is probably MAR

rf_df['use_Navigation'] = 0

rf_df = rf_df.drop(columns =['watershed_area', 'discharge_anom_top', 
                             'discharge_anom_bottom', 'use_Fisheries'])

# fill inf with nan and then remove
rf_df = rf_df.replace([np.inf, -np.inf], np.nan)
rf_df = rf_df.dropna()

### Correlation Matrix 
correlations = rf_df.iloc[:,3:51].corr()
sn.heatmap(correlations, annot=True,annot_kws={"fontsize":3.5}, xticklabels = True, yticklabels = True, cmap = 'PiYG')
plt.show()

# only big correlation
corr = rf_df.iloc[:,3:51].corr()

kot1 = corr[corr>= 0.03]
kot2 = corr[corr< -0.03]
sn.heatmap(kot1, xticklabels = True, yticklabels = True,cmap="flare", annot= True,annot_kws={"fontsize":5.9})

sn.heatmap(kot2,xticklabels = True, yticklabels = True, cmap="viridis", annot= True,annot_kws={"fontsize":5.9})
plt.show()

## RF test with sklearn  
labels = rf_df.iloc[:, 0:10] # 10 param model
labels_np = np.array(labels)

# 10 param
features = rf_df.iloc[:, 12:48 ]




###### ALL 6 Params #####

train_features, test_features, train_labels, test_labels = train_test_split(np.array(features), np.array(labels_np), 
                                                                            test_size = 0.25, random_state = 42)

rf = RandomForestRegressor(n_estimators = 100, random_state = 42,oob_score = True )
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
for pair in feature_importances:
    logger.info('Variable: %-20s Importance: %s', *pair)

## Repeat for low curves
# Calculate mean absolute percentage error (MAPE)
# add in stdev and and rmse
rmse_all = np.sqrt(np.mean((predictions-test_labels)**2))
rmse1 = np.sqrt(np.mean((predictions[:,0]-test_labels[:,0])**2))
rmse2 = np.sqrt(np.mean((predictions[:,1]-test_labels[:,1])**2))
rmse3 = np.sqrt(np.mean((predictions[:,2]-test_labels[:,2])**2))
nrmse_all = np.sqrt(rmse1/np.mean(test_labels[:,0])**2 + rmse2/np.mean(test_labels[:,1])**2 + rmse3/np.mean(test_labels[:,2])**2)

stdev = np.std(predictions)
logger.info("RMSE (all): %.4f, NRMSE: %.4f, StdDev: %.4f", rmse_all, nrmse_all, stdev)


# square plot

# Scatter plots of RF predicted vs observed STARFIT parameters
# Each subplot shows one parameter — points on the diagonal = perfect prediction
# Labels ordering confirmed from labels.columns:
# 0: flood_p1, 1: flood_p2, 2: flood_p3, 3: max_flood, 4: min_flood,
# 5: conserve_p1, 6: conserve_p2, 7: conserve_p3, 8: conserve_max, 9: conserve_min

param_names = [
    'flood_p1 (alpha)', 'flood_p2 (beta)', 'flood_p3 (mu)', 'max_flood', 'min_flood',
    'conserve_p1 (alpha)', 'conserve_p2 (beta)', 'conserve_p3 (mu)', 'conserve_max', 'conserve_min'
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Random Forest: Predicted vs Observed STARFIT Parameters', fontsize=14)

for i, ax in enumerate(axes.flat):
    pred = predictions[:, i]
    obs = test_labels[:, i]

    ax.scatter(pred, obs, s=10, alpha=0.6)

    # dynamic limits based on actual data range with a small margin
    margin = (max(pred.max(), obs.max()) - min(pred.min(), obs.min())) * 0.05
    lim_min = min(pred.min(), obs.min()) - margin
    lim_max = max(pred.max(), obs.max()) + margin
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # 1:1 diagonal
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.8)

    ax.set_title(param_names[i])
    ax.set_xlabel('Predicted')
    if i % 5 == 0:  # only left column gets y label
        ax.set_ylabel('Observed')

plt.tight_layout()
plt.show()





raw_data_file = 'Data/RF/Turner_resultsResOps/'
#test_values = '9331.csv' # 1526.csv


input_values ='Data/RF/random_forest_inputs_correct.csv'
input_data = pd.read_csv(input_values)
input_data = input_data.drop(columns =['watershed_area', 'discharge_anom_top', 'discharge_anom_bottom',
                             'use_Other', 'use_Recreation', 'use_Water Supply'])
input_data['use_Navigation'] = 0


def plot_dam(ax, dam_id, rf_model, input_df, turner_data_path, feature_columns):
    # load turner observed curves for this dam
    turner_data = pd.read_csv(turner_data_path + f'{dam_id}.csv')
    
    # get dam features and fix typo
    index_dam = input_df[input_df['hydrolakes'] == dam_id]
    index_dam = index_dam.rename(columns={'dome_spring': 'dom_spring'})
    
    # predict all 10 STARFIT parameters
    all_variables = rf_model.predict(index_dam[feature_columns])
    flood_p1, flood_p2, flood_p3 = all_variables[0,0], all_variables[0,1], all_variables[0,2]
    flood_max, flood_min = all_variables[0,3], all_variables[0,4]
    con_p1, con_p2, con_p3 = all_variables[0,5], all_variables[0,6], all_variables[0,7]
    con_max, con_min = all_variables[0,8], all_variables[0,9]

    # reconstruct sinusoidal curves from predicted parameters
    curve_flood = create_sinusodial_curve_constrained(flood_p1, flood_p2, flood_p3, flood_max, flood_min, 'flood')
    curve_conserve = create_sinusodial_curve_constrained(con_p1, con_p2, con_p3, con_max, con_min, 'conserve')

    # compute RMSE against turner observed curves
    rmse_flood = np.sqrt(np.mean((curve_flood['flood'] - turner_data['flood'])**2))
    rmse_conserve = np.sqrt(np.mean((curve_conserve['conserve'] - turner_data['conservation'])**2))

    # plot on the provided axis
    ax.plot(turner_data['epiweek'], turner_data['flood'], color='navy', label='Turner flood')
    ax.plot(turner_data['epiweek'], turner_data['conservation'], color='red', label='Turner conservation')
    ax.plot(curve_flood['epiweek'], curve_flood['flood'], color='lightblue', label='RF flood')
    ax.plot(curve_conserve['epiweek'], curve_conserve['conserve'], color='pink', label='RF conservation')
    ax.legend(loc='upper right', fontsize=6)
    ax.set_title(f'Dam {dam_id}\nRMSE flood: {rmse_flood:.1f}, conserve: {rmse_conserve:.1f}', fontsize=8)
    ax.set_xlabel('Epiweek')
    ax.set_ylabel('Storage %')

# plot all dams in one figure
dam_ids = [101498, 113172, 111866, 15701, 9748]
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('RF vs Turner STARFIT Curves per Dam', fontsize=14)

for ax, dam_id in zip(axes, dam_ids):
    plot_dam(ax, dam_id, rf, input_data, raw_data_file, features.columns)

plt.tight_layout()
plt.show()
## Test 1:


# ### Model test
# raw_data_file = 'Data/RF/Turner_resultsResOps/'
# test_values = '9331.csv' # 1526.csv


# input_values ='Data/RF/random_forest_inputs_correct.csv'
# input_data = pd.read_csv(input_values)
# input_data = input_data.drop(columns =['watershed_area', 'discharge_anom_top', 'discharge_anom_bottom',
#                              'use_Other', 'use_Recreation', 'use_Water Supply'])
# input_data['use_Navigation'] = 0
# #### WATER SUPPLY
# #### FLOOD CONTROL
# # dam index 200
# file_0 = raw_data_file + '101498.csv'

# turner_data = pd.read_csv(file_0)
# index_dam = input_data[input_data['hydrolakes'] == 101498]


# index_dam = index_dam.rename(columns={'dome_spring': 'dom_spring'})
# all_variables = rf.predict(index_dam[features.columns])
# alpha = all_variables[0,0]
# beta = all_variables[0,1]
# mu = all_variables[0,2]
# flood_max = all_variables[0,3]
# flood_min = all_variables [0,4]

# a_low = all_variables[0,5]
# b_low = all_variables[0,6]
# mu_low = all_variables[0,7]
# conserve_max = all_variables[0,8]
# conserve_min = all_variables[0,9]

# curve_flood= create_sinusodial_curve_constrained(alpha, beta, mu, flood_max, flood_min,"flood")
# curve_conserve = create_sinusodial_curve_constrained(a_low, b_low, mu_low, conserve_max, conserve_min,"conserve")

# rmse_101498_high = np.sqrt(np.mean((curve_flood['flood']- turner_data['flood'])**2))
# rmse_101498_low = np.sqrt(np.mean((curve_conserve['conserve']- turner_data['conservation'])**2))
# print('RMSE flood is ' + str(rmse_101498_high) + ' and RMSE conservation is ' + str(rmse_101498_low))


# # plot against one another: Dam index 5
# plt.plot(turner_data['epiweek'], turner_data['flood'], color = 'navy', label='turner observed flood curve  ')
# plt.plot(curve_flood['epiweek'], curve_flood['flood'], color = 'lightblue', label = 'RF flood curve')
# plt.plot(curve_conserve['epiweek'], curve_conserve['conserve'], color = 'pink', label = 'RF conservation curve')
# plt.plot(turner_data['epiweek'], turner_data['conservation'], color ='red', label = 'turner observed conservation curve')
# plt.legend()
# plt.title("RF and Turner for Dam 101498")
# plt.show()

# #### FLOOD CONTROL
# # dam index 200
# file_29 = raw_data_file + '113172.csv'
# turner_data = pd.read_csv(file_29)
# index_dam = input[input['hydrolakes'] == 113172]



# index_dam = index_dam.rename(columns={'dome_spring': 'dom_spring'})
# all_variables = rf.predict(index_dam[features.columns])
# alpha = all_variables[0,0]
# beta = all_variables[0,1]
# mu = all_variables[0,2]

# a_low = all_variables[0,3]
# b_low = all_variables[0,4]
# mu_low = all_variables[0,5]


# curve_flood= create_sinusodial_curve(alpha, beta, mu,"flood")
# curve_conserve = create_sinusodial_curve(a_low, b_low, mu_low, "conserve")
# rmse_101498_high = np.sqrt(np.mean((curve_flood['flood']- turner_data['flood'])**2))
# rmse_101498_low = np.sqrt(np.mean((curve_conserve['conserve']- turner_data['conservation'])**2))
# print('RMSE flood is ' + str(rmse_101498_high) + ' and RMSE conservation is ' + str(rmse_101498_low))

# # plot against one another: Dam index 5
# plt.plot(turner_data['epiweek'], turner_data['flood'], color = 'navy', label = 'turner observed flood curve  ')
# plt.plot(curve_flood['epiweek'], curve_flood['flood'], color = 'lightblue', label = 'RF flood curve')
# plt.plot(curve_conserve['epiweek'], curve_conserve['conserve'], color = 'pink', label = 'RF conservation curve')
# plt.plot(turner_data['epiweek'], turner_data['conservation'], color ='red', label = 'turner observed conservation curve')
# plt.legend()
# plt.title("RF and Turner for Dam 113172")
# plt.show()

# ## FC 
# index_dam = input[input['hydrolakes'] == 111866.0]
# turner_data = pd.read_csv(raw_data_file +'111866.csv')
# index_dam = index_dam.rename(columns={'dome_spring': 'dom_spring'})
# all_variables = rf.predict(index_dam[features.columns])
# alpha = all_variables[0,0]
# beta = all_variables[0,1]
# mu = all_variables[0,2]

# a_low = all_variables[0,3]
# b_low = all_variables[0,4]
# mu_low = all_variables[0,5]
# curve_flood= create_sinusodial_curve(alpha, beta, mu,"flood")
# curve_conserve = create_sinusodial_curve(a_low, b_low, mu_low, "conserve")

# rmse_101498_high = np.sqrt(np.mean((curve_flood['flood']- turner_data['flood'])**2))
# rmse_101498_low = np.sqrt(np.mean((curve_conserve['conserve']- turner_data['conservation'])**2))
# print('RMSE flood is ' + str(rmse_101498_high) + ' and RMSE conservation is ' + str(rmse_101498_low))
# # plot against one another: Dam index 5
# plt.plot(turner_data['epiweek'], turner_data['flood'], color = 'navy', label = 'turner observed flood curve  ')
# plt.plot(curve_flood['epiweek'], curve_flood['flood'], color = 'lightblue', label = 'RF flood curve')
# plt.plot(curve_conserve['epiweek'], curve_conserve['conserve'], color = 'pink', label = 'RF conservation curve')
# plt.plot(turner_data['epiweek'], turner_data['conservation'], color ='red', label = 'turner observed conservation curve')
# plt.legend()
# plt.title("RF and Turner for Dam 111866")
# plt.show()

# # dam index 200
# file_200 = raw_data_file + '15701.csv'
# index_dam = input[input['hydrolakes'] == 15701]

# turner_data = pd.read_csv(file_200)

# features_dam = pd.DataFrame(index_dam)
# index_dam = index_dam.rename(columns={'dome_spring': 'dom_spring'})
# all_variables = rf.predict(index_dam[features.columns])
# alpha = all_variables[0,0]
# beta = all_variables[0,1]
# mu = all_variables[0,2]

# a_low = all_variables[0,3]
# b_low = all_variables[0,4]
# mu_low = all_variables[0,5]
# curve_flood= create_sinusodial_curve(alpha, beta, mu,"flood")
# curve_conserve = create_sinusodial_curve(a_low, b_low, mu_low, "conserve")

# rmse_101498_high = np.sqrt(np.mean((curve_flood['flood']- turner_data['flood'])**2))
# rmse_101498_low = np.sqrt(np.mean((curve_conserve['conserve']- turner_data['conservation'])**2))
# print('RMSE flood is ' + str(rmse_101498_high) + ' and RMSE conservation is ' + str(rmse_101498_low))
# # plot against one another: Dam index 5
# plt.plot(turner_data['epiweek'], turner_data['flood'], color = 'navy', label = 'turner observed flood curve  ')
# plt.plot(curve_flood['epiweek'], curve_flood['flood'], color = 'lightblue', label = 'RF flood curve')
# plt.plot(curve_conserve['epiweek'], curve_conserve['conserve'], color = 'pink', label = 'RF conservation curve')
# plt.plot(turner_data['epiweek'], turner_data['conservation'], color ='red', label = 'turner observed conservation curve')
# plt.legend()
# plt.title("RF and Turner for Dam 15701")
# plt.show()


# ## DAM 539

# # dam index 200
# file_539 = raw_data_file + '9748.csv'
# turner_data = pd.read_csv(file_539)

# # this doens't work
# index_dam = input[input['hydrolakes'] == 9748]
# index_dam = index_dam.rename(columns={'dome_spring': 'dom_spring'})
# all_variables = rf.predict(index_dam[features.columns])
# alpha = all_variables[0,0]
# beta = all_variables[0,1]
# mu = all_variables[0,2]

# a_low = all_variables[0,3]
# b_low = all_variables[0,4]
# mu_low = all_variables[0,5]
# curve_flood= create_sinusodial_curve(alpha, beta, mu,"flood")
# curve_conserve = create_sinusodial_curve(a_low, b_low, mu_low, "conserve")
# rmse_101498_high = np.sqrt(np.mean((curve_flood['flood']- turner_data['flood'])**2))
# rmse_101498_low = np.sqrt(np.mean((curve_conserve['conserve']- turner_data['conservation'])**2))
# print('RMSE flood is ' + str(rmse_101498_high) + ' and RMSE conservation is ' + str(rmse_101498_low))
# # plot against one another: Dam index 5
# plt.plot(turner_data['epiweek'], turner_data['flood'], color = 'navy', label = 'turner observed flood curve  ')
# plt.plot(curve_flood['epiweek'], curve_flood['flood'], color = 'lightblue', label = 'RF flood curve')
# plt.plot(curve_conserve['epiweek'], curve_conserve['conserve'], color = 'pink', label = 'RF conservation curve')
# plt.plot(turner_data['epiweek'], turner_data['conservation'], color ='red', label = 'turner observed conservation curve')
# plt.legend()
# plt.title("RF and Turner for Dam 9748")
# plt.show()


'''
#### EXTRA #########



#### SINGLE VARIABLE STUFF
base_features = new_features.iloc[:,5:-1]
#features_alpha = pd.concat([new_features['alpha_high'], base_features],axis =1)
#features_beta = pd.concat([new_features['beta_high'], base_features],axis =1)
#features_mu = pd.concat([new_features['mu_high'], base_features],axis =1)

labels_alpha = new_features['alpha_high']
labels_beta = new_features['beta_high']
labels_mu = new_features['mu_high']



afeat_train, afeat_test, alab_train, alab_test = train_test_split(np.array(base_features),np.array(labels_alpha), 
                                                                  test_size = .25, random_state=0)


bfeat_train, bfeat_test, blab_train, blab_test = train_test_split(np.array(base_features),np.array(labels_beta), 
                                                                  test_size = .25, random_state=0)


ufeat_train, ufeat_test, ulab_train, ulab_test = train_test_split(np.array(base_features),np.array(labels_mu), 
                                                                  test_size = .25, random_state=0)
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# alpha
reg_a = GradientBoostingRegressor(random_state=0)

reg_a.fit(afeat_train, alab_train)
predict_a =reg_a.predict(afeat_test)
rmse_alpha =  np.sqrt(np.mean((predict_a - alab_test)**2))
plt.scatter(predict_a,  alab_test, color='blue')
plt.title('Alpha Scatter Grad Boost')

# beta
reg_b = GradientBoostingRegressor(random_state=0)

reg_b.fit(bfeat_train, blab_train)
predict_b =reg_b.predict(bfeat_test)
rmse_beta =  np.sqrt(np.mean((predict_b - blab_test)**2))
plt.scatter(predict_b,  blab_test, color ='orange')
plt.title("Beta Scatter Grad Boost")

# mu
reg_u = GradientBoostingRegressor(random_state=0)

reg_u.fit(ufeat_train, ulab_train)
predict_u =reg_u.predict(ufeat_test)
rmse_mu =  np.sqrt(np.mean((predict_u - ulab_test)**2))
plt.scatter(predict_u,  ulab_test, color ='green')
plt.title('MU Scatter Grad Boost')

## SINGLE RF PER VAR
# alpha
rf_a =RandomForestRegressor(n_estimators = 100, random_state = 42,
                           oob_score = False, bootstrap = True, )

rf_a.fit(afeat_train, alab_train)
predict_a =rf_a.predict(afeat_test)
rmse_alpha_rf =  np.sqrt(np.mean((predict_a - alab_test)**2))
plt.scatter(predict_a,  alab_test)
plt.title("Alpha RF Scatter")


# beta
rf_b =RandomForestRegressor(n_estimators = 100, random_state = 42,
                           oob_score = False, bootstrap = True, )

rf_b.fit(bfeat_train, blab_train)
predict_b =rf_b.predict(bfeat_test)
rmse_beta_rf =  np.sqrt(np.mean((predict_b - blab_test)**2))
plt.scatter(predict_b,  blab_test, color ='orange')
plt.title("Beta RF Scatter")

# mu
rf_u =RandomForestRegressor(n_estimators = 100, random_state = 42,
                           oob_score = False, bootstrap = True, )

rf_u.fit(ufeat_train, ulab_train)
predict_u =rf_u.predict(ufeat_test)
rmse_mu_rf =  np.sqrt(np.mean((predict_u - ulab_test)**2))
plt.scatter(predict_u,  ulab_test, color ='green')
plt.title("Mu Scatter RF")



#plt.plot(input['epiweek'], input['raw_flood'])

#### RF ###
alpha_rf = rf_a.predict(index_dam.iloc[:,6:-1])
beta_rf = rf_b.predict(index_dam.iloc[:,6:-1])
mu_rf = rf_u.predict(index_dam.iloc[:,6:-1])


curve_rf= create_sinusodial_curve(alpha_rf, beta_rf, mu_rf,"flood")

#=#### Grad Boost ###
alpha_gb = reg_a.predict(index_dam.iloc[:,6:-1])
beta_gb = reg_b.predict(index_dam.iloc[:,6:-1])
mu_gb = reg_u.predict(index_dam.iloc[:,6:-1])

curve_xg= create_sinusodial_curve(alpha_gb, beta_gb, mu_gb,"flood")


# plot against one another:
plt.plot(turner_data['epiweek'], turner_data['flood'], color = 'black')
plt.plot(curve_rf['epiweek'], curve_rf['flood'], color = 'pink')
plt.plot(curve_xg['epiweek'], curve_xg['flood'], color = 'purple')
plt.legend()
plt.title("Rf, XG and Turner")
## Combination of all varialbes for RF ####


# oob_scorebool or callable, default=False set to TRUE will give rsquared 
rf = RandomForestRegressor(n_estimators = 100, random_state = 42,
                           oob_score = my_scorer, bootstrap = True )

rf.fit(train_features, train_labels)

#joblib.dump(rf,'/home/steya001/res_ops/random_forest.joblib')
predictions = rf.predict(test_features)

rf.score
errors = np.sqrt(np.mean(abs(predictions - test_labels)**2))

print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
# add in stdev and and rmse
rmse_all = np.sqrt(np.mean((predictions-test_labels)**2))
rmse1 = np.sqrt(np.mean((predictions[0]-test_labels[0])**2))
rmse2 = np.sqrt(np.mean((predictions[1]-test_labels[1])**2))
rmse3 = np.sqrt(np.mean((predictions[2]-test_labels[2])**2))
stdev = np.std(predictions)

mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

from scipy.stats import spearmanr
spearman_r_0 = spearmanr(predictions[0], test_labels[0])
spearman_r_1 = spearmanr(predictions[1], test_labels[1])
spearman_r_2 = spearmanr(predictions[2], test_labels[2])
spearman_all, p = spearmanr(predictions, test_labels)

pd.DataFrame(spearman_all).shape
pd.DataFrame(p)

plt.plot(predictions, test_labels)
plt.scatter(predictions[:,0], test_labels[:,0], color = 'blue' )
plt.scatter(predictions[:,1], test_labels[:,1], color = 'orange' )
plt.scatter(predictions[:,2], test_labels[:,2], color = 'green' )


plot_alpha = pd.DataFrame({'alpha_pred': predictions[:,1], 
                           'alpha_labels' :test_labels[:,1]})
plot_alpha = plot_alpha[plot_alpha['alpha_labels'] >-40]
plot_alpha = plot_alpha[plot_alpha['alpha_pred'] >-35]
plt.scatter(plot_alpha['alpha_pred'], plot_alpha['alpha_labels'], color = 'orange' )


plot_beta = pd.DataFrame({'beta_pred': predictions[:,2], 
                           'beta_labels' :test_labels[:,2]})
plot_beta = plot_beta[plot_beta['beta_labels'] <40]
#plot_beta= plot_beta[plot_beta['beta_pred'] <20]

plt.scatter(plot_beta['beta_pred'], plot_beta['beta_labels'], color = 'green' )


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
## Repeat for low curves

# rf_df _low
label_low = pd.DataFrame()
label_low['alpha_low'] = params['NORlo_alpha']
label_low['beta_low'] = params['NORlo_beta']
label_low['mu_low'] = params['NORlo_max']
label_low['hydrolakes'] = params['hydrolakes_id']

merged_labels_low = pd.merge(rf_df.iloc[:,1:9], label_low, on = 'hydrolakes')
final_merged = pd.merge(merged_labels_low, inflow_new, on= 'hydrolakes')
clean_df =final_merged[ np.isneginf(final_merged['mu_low']) == False]

clean_df = clean_df.iloc[:,4:-1]

labels_low_np = np.array(clean_df.iloc[:,4:7])

features_low = clean_df.iloc[:,0:4]
features_low = pd.concat([features_low,clean_df.iloc[:,8:-1]], axis = 1)
 

features_low[np.isnan(features_low) == True] = 0   

feature_list_low = list(features_low.columns)
features_low_np = np.array(features_low)


# now I can split to train and test data
train_features_low, test_features_low, train_labels_low, test_labels_low = train_test_split(features_low_np, labels_low_np, 
                                                                            test_size = 0.25, random_state = 42)





rf_low = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf_low.fit(train_features_low, train_labels_low)
predictions_low = rf_low.predict(test_features_low)

errors_low = abs(predictions_low - test_labels_low)

print('Mean Absolute Error:', round(np.mean(errors_low), 2))

# Calculate mean absolute percentage error (MAPE)
mape_low = 100 * (errors_low / test_labels_low)
# Calculate and display accuracy
accuracy_low = 100 - np.mean(mape_low)
print('Accuracy:', round(accuracy_low, 2), '%.')


# Calculate mean absolute percentage error (MAPE)
# add in stdev and and rmse
rmse_all_low = np.sqrt(np.mean((predictions_low-test_labels_low)**2))
rmse1_low = np.sqrt(np.mean((predictions_low[0]-test_labels_low[0])**2))
rmse2_low = np.sqrt(np.mean((predictions_low[1]-test_labels_low[1])**2))
rmse3_low = np.sqrt(np.mean((predictions_low[2]-test_labels_low[2])**2))
stdev_low = np.std(predictions_low)



plt.plot(predictions_low, test_labels_low)
plt.scatter(predictions_low[:,0], test_labels_low[:,0], color = 'blue' )
plt.scatter(predictions_low[:,1], test_labels_low[:,1], color = 'orange' )
plt.scatter(predictions_low[:,2], test_labels_low[:,2], color = 'green' )
'''