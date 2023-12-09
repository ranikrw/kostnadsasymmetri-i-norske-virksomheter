import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import statsmodels.api as sm

import os

# Inkluderer funksjoner fra filen "funksjoner.py"
from funksjoner import *

##############################################
## Definer hvilke kostnader som skal analyseres
##############################################
costs_for_response = 'Varekostnader'

# Mulige verdier:
# 'Driftskostnader'
# 'Varekostnader'

if (costs_for_response != 'Driftskostnader') & (costs_for_response != 'Varekostnader'):
    raise Exception("Feil: costs_for_response må være 'Driftskostnader' eller 'Varekostnader'")

##############################################
## Lager data frames for å sette inn resultater
##############################################
results_df = pd.DataFrame()
results_df.index.name = costs_for_response

sample_selection_table = pd.DataFrame()

##############################################
## Laster data som ble behandlet i 
## "behandle_data_og_lag_variabler.py"
##############################################
data_all = pd.read_csv('../data_behandlet/data_behandlet.csv',sep=';',low_memory=False)

# De aller minste bedriftene har begrenset med kostnader knyttet til ansatte og 
# varige eiendeler. Vi begrenser derfor vårt utvalg til bedrifter med 
# lønnskostnader over kroner 5 millioner.
data_all = data_all[data_all['Lonnskostnader_ikke_deflatert']>5e6].reset_index(drop=True)

# Sjekker at alle observasjoner er unike firma-år
unique_orgnr = data_all.groupby(['orgnr','regnaar']).size().reset_index()
temp = unique_orgnr[0].unique()
if len(temp)!=1:
    raise Exception("Feil: ikke alle firma-år er unike")

##############################################
##  Modell 1
##############################################
file_name   = '1'

# Antall tidligere regnskapsår som blir brukt i regresjon
num_prev = 1

# Sample selection
var_log = [
    costs_for_response,
    costs_for_response+'_prev',
    'Salg',
    'Salg_prev',
]
data,sample_selection_table = sample_selection(data_all,num_prev,var_log,file_name,sample_selection_table,costs_for_response)

# Lager variabler for regresjon
data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))
data['lnSalg']     = np.log((data['Salg']/data['Salg_prev']).astype(float))
data['DlnSalg']    = (data['lnSalg']<0)*data['lnSalg']

# Definerer hvilke variabler som skal brukes i regresjon
var = [
    'lnCost_'+costs_for_response,
    'lnSalg',
    'DlnSalg',
]

# Gjennomfører regresjon og lagrer resultater i results_df
results_df = regression_rrw(var,data,results_df,file_name)


##############################################
##  Modell 2
##############################################
file_name   = '2'

# Antall tidligere regnskapsår som blir brukt i regresjon
num_prev = 1

# Sample selection
var_log = [
    costs_for_response,
    costs_for_response+'_prev',
    'Salg',
    'Salg_prev',
    'Eiendeler',
    'Lonnskostnader',
]
data,sample_selection_table = sample_selection(data_all,num_prev,var_log,file_name,sample_selection_table,costs_for_response)

# Lager variabler for regresjon
data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))
data['lnSalg']     = np.log((data['Salg']/data['Salg_prev']).astype(float))
data['DlnSalg']    = (data['lnSalg']<0)*data['lnSalg']

data['EINT']    = np.log((data['Eiendeler']/data['Salg']).astype(float))
data['AINT']    = np.log((data['Lonnskostnader']/data['Eiendeler']).astype(float))
data['BNP']     = (data['bnp']/data['bnp_prev']).astype(float)-1

data['lnSalgEINT'] = data['lnSalg']*data['EINT']
data['lnSalgAINT'] = data['lnSalg']*data['AINT']
data['lnSalgBNP']  = data['lnSalg']*data['BNP']

data['DlnSalgEINT'] = data['DlnSalg']*data['EINT']
data['DlnSalgAINT'] = data['DlnSalg']*data['AINT']
data['DlnSalgBNP']  = data['DlnSalg']*data['BNP']


# Definerer hvilke variabler som skal brukes i regresjon
var = [
    'lnCost_'+costs_for_response,
    'lnSalg',
    'DlnSalg',
    'lnSalgEINT',
    'lnSalgAINT',
    'lnSalgBNP',
    'DlnSalgEINT',
    'DlnSalgAINT',
    'DlnSalgBNP',
]

# Gjennomfører regresjon og lagrer resultater i results_df
results_df = regression_rrw(var,data,results_df,file_name)


##############################################
##  Modell 3
##############################################
file_name   = '3'

# Antall tidligere regnskapsår som blir brukt i regresjon
num_prev = 2

# Sample selection
var_log = [
    costs_for_response,
    costs_for_response+'_prev',
    'Salg',
    'Salg_prev',
    'Salg_prev_prev',
]
data,sample_selection_table = sample_selection(data_all,num_prev,var_log,file_name,sample_selection_table,costs_for_response)

# Lager variabler for regresjon
data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))
data['lnSalg']     = np.log((data['Salg']/data['Salg_prev']).astype(float))
data['DlnSalg']    = (data['lnSalg']<0)*data['lnSalg']
data['lnSalgPrev']     = np.log((data['Salg_prev']/data['Salg_prev_prev']).astype(float))
data['DlnSalgPrev']    = (data['lnSalgPrev']<0)*data['lnSalg']


# Definerer hvilke variabler som skal brukes i regresjon
var = [
    'lnCost_'+costs_for_response,
    'lnSalg',
    'DlnSalg',
    'lnSalgPrev',
    'DlnSalgPrev',
]

# Gjennomfører regresjon og lagrer resultater i results_df
results_df = regression_rrw(var,data,results_df,file_name)


##############################################
##  Modell 4
##############################################
file_name   = '4'

# Antall tidligere regnskapsår som blir brukt i regresjon
num_prev = 2

# Sample selection
var_log = [
    costs_for_response,
    costs_for_response+'_prev',
    'Salg',
    'Salg_prev',
    'Salg_prev_prev',
]
data,sample_selection_table = sample_selection(data_all,num_prev,var_log,file_name,sample_selection_table,costs_for_response)

# Lager variabler for regresjon
data['lnCost_'+costs_for_response]  = np.log((data[costs_for_response]/data[costs_for_response+'_prev']).astype(float))

data['lnSalg']     = np.log((data['Salg']/data['Salg_prev']).astype(float))
data['DlnSalg']    = (data['lnSalg']<0)*data['lnSalg']

data['lnSalgPrev'] = np.log((data['Salg_prev']/data['Salg_prev_prev']).astype(float))
data['I_prev']      = (data['lnSalgPrev']>0)
data['D_prev']      = (data['lnSalgPrev']<0)

data['I_prev_lnSalg']    = data['I_prev']*data['lnSalg']
data['I_prev_DlnSalg']    = data['I_prev']*data['DlnSalg']

data['D_prev_lnSalg']    = data['D_prev']*data['lnSalg']
data['D_prev_DlnSalg']    = data['D_prev']*data['DlnSalg']

# Definerer hvilke variabler som skal brukes i regresjon
var = [
    'lnCost_'+costs_for_response,
    'I_prev_lnSalg',
    'I_prev_DlnSalg',
    'D_prev_lnSalg',
    'D_prev_DlnSalg',
]

# Gjennomfører regresjon og lagrer resultater i results_df
results_df = regression_rrw(var,data,results_df,file_name)

##############################################
## Omorganiserer resultattabellen
##############################################
results_df = shift_row_to_bottom('Konstant',results_df)
results_df = shift_row_to_bottom('År FE',results_df)
results_df = shift_row_to_bottom('Bransje FE',results_df)
results_df = shift_row_to_bottom('R2',results_df)
results_df = shift_row_to_bottom('Antall observasjoner',results_df)

string_for_save = '_'+costs_for_response
results_df.index.name = string_for_save

##############################################
##  Lagrer resultater i mappen 'resutater'
##############################################
folder_name = 'resultater/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

results_df.to_excel(folder_name+'Results'+string_for_save+'.xlsx')
sample_selection_table.to_excel(folder_name+'Sample_selection'+string_for_save+'.xlsx')

