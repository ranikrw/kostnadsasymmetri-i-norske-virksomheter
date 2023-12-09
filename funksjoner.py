import numpy as np
import pandas as pd

from statsmodels.formula.api import ols

import os

from scipy.stats import pearsonr

from sklearn import preprocessing

def add_tailing_zeros_decimals(num,num_decimals):
    while len(num[num.rfind('.')+1:])!=num_decimals:
        num = num+'0'
    return num

# https://stackoverflow.com/questions/67113820/function-to-move-specific-row-to-top-or-bottom-of-pandas-dataframe
def shift_row_to_bottom(col_to_shift,df):
  idx = df.index.tolist()
  idx.remove(col_to_shift)
  df = df.reindex(idx + [col_to_shift])
  return df

def model_preparing(X,y,data):
    df = pd.concat([X,data['orgnr']],axis=1)
    df = pd.concat([df,y],axis=1)
    for i in X.columns:
        if i == X.columns[0]:
            string_formula = str(y.name) +' ~ '+i
        else:
            string_formula = string_formula+' + '+str(i)
    return df,string_formula

def regression_rrw(var,data,results_df,file_name):

    # Sett til 'True'/'False' for å inkludere/ikke inkludere 
    # faste effekter av regnskapsår 
    fixed_effects_year = True

    # Sett til 'True'/'False' for å inkludere/ikke inkludere 
    # faste effekter av bransje 
    fixed_effects_Bransje = True

    # Antall desimaler etter komma i tabellene med resultater
    num_decimals = 3

    y = data[var[0]]
    X = data[var[1:]]

    # Add fixed effects
    if fixed_effects_year:
        temp = pd.get_dummies(data['regnaar'].astype(int)).iloc[:,:-1]
        for i in temp.columns:
            temp=temp.rename(columns = {i:'dy'+str(i)})
        X = pd.concat([X,temp],axis=1)
    if fixed_effects_Bransje:
        temp = pd.get_dummies(data['Bransje']).iloc[:,:-1]
        for i in temp.columns:
            temp=temp.rename(columns = {i:'di'+str(i)})
        X = pd.concat([X,temp],axis=1)

    df,string_formula = model_preparing(X,y,data)

    model = ols(formula=string_formula,data=df).fit(cov_type='cluster', cov_kwds={'groups': df['orgnr']})

    list_variables = ['Intercept'] + var[1:]
    series_results     = pd.Series(dtype=object)
    for i in list_variables:
        coef = np.round(model.params[i],num_decimals).astype(str)
        coef = add_tailing_zeros_decimals(coef,num_decimals)
        coef = coef.replace('.', ',')
        
        pval = model.pvalues[i]
        if pval<0.001:
            coef = coef+'****'
        elif pval<0.01:
            coef = coef+'***'
        elif pval<0.05:
            coef = coef+'**'
        elif pval<0.10:
            coef = coef+'*'

        if i == 'Intercept':
            series_results['Konstant'] = coef
        else:
            series_results[i] = coef

    # Antall observasjoner
    series_results['Antall observasjoner'] = thousand_seperator(X.shape[0])

    # R squared
    series_results['R2'] = np.round(model.rsquared,num_decimals).astype(str).replace('.', ',')

    # Fixed effects
    if fixed_effects_year:
        series_results['År FE']  = 'Ja'
    else:
        series_results['År FE']  = 'Nei'
    if fixed_effects_Bransje:
        series_results['Bransje FE']  = 'Ja'
    else:
        series_results['Bransje FE']  = 'Nei'

    # Legger til regresjonsresultatene til tabell med 
    # resultater fra andre modeller
    series_results.name = 'Modell ('+file_name+')'

    results_df = pd.concat([results_df,series_results],axis=1)

    return results_df


def exclude_missing_prev_year(num_prev,data,sample_selection_series):
    col_name_1 = 'Ingen årsregnskap det foregående året'
    col_name_2 = 'Ingen årsregnskap de to foregående årene'
    if (num_prev==1)|(num_prev==2):
        ind = pd.isnull(data['Salg_prev'])==False
        data = data[ind]
        data = data.reset_index(drop=True) # Reset index
        sample_selection_series[col_name_1] = thousand_seperator(np.sum(ind==False))
    if num_prev==2:
        ind = pd.isnull(data['Salg_prev_prev'])==False
        data = data[ind]
        data = data.reset_index(drop=True) # Reset index
        sample_selection_series[col_name_2] = thousand_seperator(np.sum(ind==False))
    else:
        sample_selection_series[col_name_2] = None
    if (num_prev!=1)&(num_prev!=2):
        print('ERROR defining num_prev in function exclude_missing_prev_year()')
    return data,sample_selection_series


def thousand_seperator(number):
    return "{:,.0f}".format(number).replace(',', ' ')


def exclude_industries(data,sample_selection_series):
    ind = data['Bransje']!='L' # L - Omsetning og drift av fast eiendom
    ind = ind & (data['Bransje']!='K') # K - Finansiering og forsikring
    ind = ind & (data['Bransje']!='O') # O - Off.adm., forsvar, sosialforsikring
    ind = ind & (data['Bransje']!='D') # D - Kraftforsyning
    ind = ind & (data['Bransje']!='E') # E - Vann, avløp, renovasjon
    ind = ind & (data['Bransje']!='0') # companies for investment and holding purposes only
    ind = ind & (data['Bransje']!='MISSING') # Missing

    data = data[ind]
    data = data.reset_index(drop=True) # Reset index 

    col_name = 'Ekskludert de nevnte bransjene'
    sample_selection_series[col_name] = thousand_seperator(np.sum(ind==False))

    return data,sample_selection_series


def removing_zero_and_negative_ratios(var_log,data,sample_selection_series):
    obs_before = data.shape[0]

    for var in var_log:
        ind = data[var]<=0
        ind = ind | (data[var].isnull())
        data = data[ind==False]
        data = data.reset_index(drop=True) # Reset index 

    col_name = 'Ikke-positiv verdi for regnskapsposter'
    sample_selection_series[col_name] = thousand_seperator(obs_before-data.shape[0])

    return data,sample_selection_series


def sample_selection(data,num_prev,var_log,file_name,sample_selection_table,costs_for_response):
    sample_selection_series = pd.Series(dtype=float)
    text_initial_sample = 'Alle årsregnskaper 2008-2021'
    sample_selection_series[text_initial_sample] = thousand_seperator(data.shape[0])

    # Exclude industries
    data,sample_selection_series = exclude_industries(data,sample_selection_series)

    # we include only firm-year observations of firms where 
    # also a firm-year from the previous accounting year is available
    data,sample_selection_series = exclude_missing_prev_year(num_prev,data,sample_selection_series)

    # Empty columns
    sample_selection_series[''] = None
    sample_selection_series[costs_for_response+':'] = None

    # Removing zero and negative
    data,sample_selection_series = removing_zero_and_negative_ratios(var_log,data,sample_selection_series)

    # Adding final sample size and merging with table
    sample_selection_series['Endelig utvalg for analyser'] = thousand_seperator(data.shape[0])
    sample_selection_table['Modell ({})'.format(file_name)] = sample_selection_series

    return data,sample_selection_table