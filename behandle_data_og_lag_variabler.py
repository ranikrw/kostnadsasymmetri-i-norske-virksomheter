import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import time
import os

from tqdm import tqdm # Showing progress bar in for-loops

##################################################################
##  Load data
##################################################################
print('-----------------------------------------')
print('Loading data:')
print('-----------------------------------------')
folder_name = '../../datasett_aarsregnskaper/data4/'
files = os.listdir(folder_name)
for current_file in files:
    file_year = int(current_file[0:4])

    # LOADING DATA
    data_loaded = pd.read_csv(folder_name+current_file,sep=';',low_memory=False)

    # Adding all data together into data
    if current_file == files[0]:
        data = data_loaded.copy()
    else:
        data = pd.concat([data,data_loaded])
    print('Imported for accounting year {}'.format(file_year))

# Reset index 
data = data.reset_index(drop=True)

# Free memory
del data_loaded

# Considering only financial year <= 2021
data = data[data.regnaar<=2021]
data = data.reset_index(drop=True) # Reset index

##################################################################
##  Lager variabler
##################################################################
# Argumentasjon for .fillna(0):
# "There are no missing values for the financial information in Appendix A. If a value is missing in 
# the dataset for any of the items listed in Appendix A, it indicates that no value was provided when 
# the financial statement was reported, that is, the value is zero."
# Wahlstrøm, R. R. (2022). Financial statements of companies in Norway. arXiv:2203.12842. https://doi.org/10.48550/arXiv.2203.12842

# Salg
data['Salg']   = data['Salgsinntekt'].fillna(0)

# Driftskostnader
data['Driftskostnader'] = data['Sum inntekter'].fillna(0)-data['Driftsresultat'].fillna(0)

# Varekostnader
data['Varekostnader']  =\
    data['Varekostnad'].fillna(0)+\
    data['Endring i beholdning av varer under tilvirkning og ferdig tilvirkede varer'].fillna(0)

# Eiendeler
data['Eiendeler']   = data['SUM EIENDELER'].fillna(0)

# Lønnskostnader
data['Lonnskostnader'] = data['Loennskostnad'].fillna(0)

# Bruttonasjonalprodukt (BNP) 
# Vi benytter deflatert BNP hentet fra tabell 09189 fra SSB: https://www.ssb.no/en/statbank/table/09189
bnp_data    = pd.read_csv('../data_BNP_KPI/GDP.csv',sep=';')
bnp_data.columns = bnp_data.columns.astype(int)
bnp_data = bnp_data.T[0]

# Bransje
data = data.rename(columns={
    'naeringskoder_level_1': 'Bransje',
    })

# Varelager
inventories = data['Varer']
ind = pd.isnull(inventories)
inventories.loc[ind] = data['Sum varer'].loc[ind]
data['Varelager']         = inventories.fillna(0) + data['Biologiske eiendeler'].fillna(0)

# Kundefordringer
kundefordringer = data['Sum fordringer']
ind = pd.isnull(kundefordringer)
kundefordringer.loc[ind] = data['Kundefordringer'].loc[ind]
data['Kundefordringer'] = kundefordringer.fillna(0)

# Leverandørgjeld
data['Leverandorgjeld']        = data['Leverandoergjeld'].fillna(0)

# Omsetning og eiendeler i EUR
data['sum_eiendeler_EUR'] = data['sum_eiendeler_EUR'].fillna(0)
data['sum_omsetning_EUR'] = data['sum_omsetning_EUR'].fillna(0)

##################################################################
##  Filtering  data
##################################################################
# Våre modeller bruker variabler som er utledet med verdier for opptil de to 
# foregående regnskapsårene. Dermed utfører vi analysene våre på 
# årsregnskaper fra og med 2008, mens årsregnskapene fra 2006 og 2007 
# kun brukes til å utlede variabelverdier.

ind = data['regnaar']>=2008

# For å analysere kun små og mellomstore bedrifter (SMB) (https://ec.europa.eu/growth/smes/sme-definition_en)
# ind = ind & ((data['sum_eiendeler_EUR'].fillna(0)<=43e6)|(data['sum_omsetning_EUR'].fillna(0)<=50e6))
# ind = ind & ((data['sum_eiendeler_EUR'].fillna(0)>2e6)&(data['sum_omsetning_EUR'].fillna(0)>2e6))

##################################################################
##  CPI: Consumer Price Index (konsumprisindeksen)
##################################################################
# For å kontrollere for inflasjon deflaterer vi alle regnskapstall 
# basert på konsumprisindeksen fra Statistisk sentralbyrå (SSB).
# Se tabell 03013 fra SSB: https://www.ssb.no/en/statbank/table/03013

CPI_data = pd.read_csv('../data_BNP_KPI/03013_20220814-033754.csv',
    sep=';',
    skiprows=2
    )
CPI_data.index.name = None
del CPI_data['consumption group']
CPI_data.rename(columns = {'Consumer Price Index (2015=100)':'CPI'}, inplace = True)

def fun_CPI(x):
    return x.replace('M','')
CPI_data['month'] = CPI_data['month'].apply(fun_CPI).astype(int)
CPI_data = CPI_data.set_index('month')
CPI_data.index.name = None
CPI_data = CPI_data['CPI']

# Avslutningsdato = Balansedato og sluttdato for regnskapets regnskapsår
data_avslutningsdato    = data['avslutningsdato']
def fun_avslutningsdato(x):
    return x[0:7].replace('-','')
data_avslutningsdato = data_avslutningsdato.apply(fun_avslutningsdato).astype(int)

data['Salg_ikke_deflatert']           = data['Salg']
data['Lonnskostnader_ikke_deflatert'] = data['Lonnskostnader']
data['Eiendeler_ikke_deflatert']      = data['Eiendeler']
columns_to_deflate = [
    'Salg',
    'Driftskostnader',
    'Varekostnader',
    'Eiendeler',
    'Lonnskostnader',
    'Leverandorgjeld',
    'Varelager',
    'Kundefordringer',
]
data = data.reset_index(drop=True)
CPIs = CPI_data.loc[data_avslutningsdato].values
for c in columns_to_deflate:
    data[c] = data[c] / CPIs

##################################################################
##  Lager variabler basert på tidligere regnskapsår
##################################################################
data['bnp']                     = None
data['bnp_prev']                = None

data_for_saving_variables = pd.DataFrame(index=data.index)
data_for_saving_variables['Salg_prev']              = None
data_for_saving_variables['Salg_prev_prev']         = None
data_for_saving_variables['Driftskostnader_prev']    = None
data_for_saving_variables['Varekostnader_prev']      = None
data_for_saving_variables['Varelager_prev']        = None

for regnaar in tqdm(data.loc[ind, 'regnaar'].unique()):
    
    # Inkluderer kun gjeldende og to foregående regnskapsår
    curr_data = data[ind & (data['regnaar'] == regnaar)]
    prev_data = data[data['regnaar'] == regnaar - 1]
    prev_prev_data = data[data['regnaar'] == regnaar - 2]
    
    # Slå sammen data på 'orgnr' for å få regnskapstall fra 
    # ett og to år tilbake i tid
    columns_prev = [
        'Salg',
        'Driftskostnader',
        'Varekostnader',
        'Varelager',
    ]
    columns_prev_prev = [
        'Salg',
    ]
    merged_data = curr_data.merge(prev_data[['orgnr']+columns_prev], on='orgnr', how='left', suffixes=('', '_prev'))
    merged_data = merged_data.merge(prev_prev_data[['orgnr']+columns_prev_prev], on='orgnr',how='left', suffixes=('', '_prev_prev'))
    merged_data.index = curr_data.index

    for c in columns_prev:
        data_for_saving_variables.loc[merged_data.index,c+'_prev'] = merged_data[c+'_prev']
    for c in columns_prev_prev:
        data_for_saving_variables.loc[merged_data.index,c+'_prev_prev'] = merged_data[c+'_prev_prev']

    data.loc[curr_data.index,'bnp']       = bnp_data.loc[regnaar]
    data.loc[curr_data.index,'bnp_prev']  = bnp_data.loc[regnaar - 1]

data = pd.concat([data,data_for_saving_variables],axis=1)

# Definerer hvilke kolonner som skal beholdes
columns_to_keep = [
    'orgnr',
    'regnaar',
    'Bransje',
    'Salg',
    'Salg_prev',
    'Salg_prev_prev',
    'Driftskostnader',
    'Driftskostnader_prev',
    'Varekostnader',
    'Varekostnader_prev',
    'Eiendeler',
    'Lonnskostnader',
    'bnp',
    'bnp_prev',
    'Leverandorgjeld',
    'Varelager',
    'Varelager_prev',
    'Kundefordringer',
    'Salg_ikke_deflatert',
    'Lonnskostnader_ikke_deflatert',
    'Eiendeler_ikke_deflatert',
    'orgform',
    'sum_eiendeler_EUR',
    'sum_omsetning_EUR',
]
data = data.loc[ind,columns_to_keep].reset_index(drop=True)

# Lagrer prossesert data fo videre analyser i filen "analysere.py"
folder_name = '../data_behandlet'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
data.to_csv(folder_name+'/data_behandlet.csv',index=False,sep=';')

