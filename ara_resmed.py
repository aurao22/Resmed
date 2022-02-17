
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\")
from ara_commons.ara_file import *
from ara_commons.ara_graph import *
from ara_commons.ara_df import *

# ---------------------------------------------------------------------------------------------
#                               CONSTANTES
# ---------------------------------------------------------------------------------------------
CSL = "CSL"
EVE = "EVE"
BRP = "BRP"
PLD = "PLD"
SAD = "SAD"

resmed_data_type = [CSL, EVE, BRP, PLD, SAD]

resmed_first_edf_file_name = "STR.edf"
resmed_day_data_dir_name = "DATALOG"


# ---------------------------------------------------------------------------------------------
#                               FONCTIONS
# ---------------------------------------------------------------------------------------------
def resmed_load_first_df(first_edf):
    resmed_first_df = _get_edf_file_df(first_edf)
    resmed_first_df['Date-Real'] = resmed_first_df['Date'].map(_define_date)

    # Transformation de l'heure
    resmed_first_df["Session duration in H"] = np.nan
    resmed_first_df["Session duration sum in H"] = np.nan
    resmed_first_df.loc[(resmed_first_df["Duration"].notna()) & (resmed_first_df["Duration"]>0), "Session duration in H"] = resmed_first_df.loc[(resmed_first_df["Duration"].notna()) &  (resmed_first_df["Duration"]>0),"Duration"].apply(lambda x: pd.Timedelta(minutes=x))
    resmed_first_df.loc[(resmed_first_df["PatientHours"].notna()) & (resmed_first_df["PatientHours"]> 0 ),"Session duration sum in H"] = resmed_first_df.loc[(resmed_first_df["PatientHours"].notna()) & (resmed_first_df["PatientHours"]> 0 ),"PatientHours"].apply(lambda x: pd.Timedelta(hours=x))
    resmed_first_df = resmed_first_df[['Date-Real', 'time', 'MaskOn', 'MaskOff', 'MaskEvents', "Session duration in H", 'Duration', "Session duration sum in H", 'PatientHours',
       'Mode', 'S.RampEnable', 'S.RampTime',
       'S.C.StartPress', 'S.C.Press', 'S.EPR.ClinEnable', 'S.EPR.EPREnable',
       'S.EPR.Level', 'S.EPR.EPRType', 'S.AS.Comfort', 'S.AS.StartPress',
       'S.AS.MaxPress', 'S.AS.MinPress', 'S.SmartStart', 'S.PtAccess',
       'S.ABFilter', 'S.Mask', 'S.Tube', 'S.ClimateControl', 'S.HumEnable',
       'S.HumLevel', 'S.TempEnable', 'S.Temp', 'HeatedTube', 'Humidifier',
       'BlowPress.95', 'BlowPress.5', 'Flow.95', 'Flow.5', 'BlowFlow.50',
       'AmbHumidity.50', 'HumTemp.50', 'HTubeTemp.50', 'HTubePow.50',
       'HumPow.50', 'SpO2.50', 'SpO2.95', 'SpO2.Max', 'SpO2Thresh',
       'MaskPress.50', 'MaskPress.95', 'MaskPress.Max', 'TgtIPAP.50',
       'TgtIPAP.95', 'TgtIPAP.Max', 'TgtEPAP.50', 'TgtEPAP.95', 'TgtEPAP.Max',
       'Leak.50', 'Leak.95', 'Leak.70', 'Leak.Max', 'MinVent.50', 'MinVent.95',
       'MinVent.Max', 'RespRate.50', 'RespRate.95', 'RespRate.Max',
       'TidVol.50', 'TidVol.95', 'TidVol.Max', 'AHI', 'HI', 'AI', 'OAI', 'CAI',
       'UAI', 'RIN', 'CSR', 'Fault.Device', 'Fault.Alarm', 'Fault.Humidifier',
       'Fault.HeatedTube', 'Crc16']]
    return resmed_first_df

def resmed_clean_df(resmed_first_df, verbose=0):
    light_df = remove_empty_numeric_columns(resmed_first_df, verbose=verbose, inplace=False)
    light_df = remove_na_columns(light_df, max_na=73, verbose=verbose, inplace=False)
    light_df = light_df[['Date-Real', 'Session duration in H', 'Session duration sum in H', 'Duration', 'MaskEvents', 
       'AHI', 'HI', 'AI', 'OAI', 'CAI', 'RIN', 'Crc16',
       'MaskPress.Max', 'Leak.95', 'AmbHumidity.50', 'Leak.50', 'MinVent.50'
       ]]

    if verbose:
        print("BEFORE:", light_df.shape, end="")
    light_df = light_df.drop_duplicates(light_df.columns, keep='first')
    if verbose:
        print("=>",light_df.shape)
    return light_df


def resmed_get_annotations_df(datas_path, dataset_collecting_date, verbose=0):
    annot_df = _get_annotations_by_date(datas_path, dataset_collecting_date)
    annot_df = annot_df[['date_time', 'type', 'description', 'duration', 'onset', 'day']]
    annot_df["onset"] = annot_df['onset'].apply(lambda x: x.to_pydatetime() - datetime(1970, 1, 1, 0, 0))
    annot_df["date_time_event"] = annot_df["date_time"] + annot_df['onset']
    annot_df = annot_df[['date_time', "date_time_event", 'type', 'description', 'duration', 'onset', 'day']]
    annot_df["description-fr"] = annot_df["description"]
    annot_df["Apnee"] = False
    annot_df["code"] = np.nan

    annot_df.loc[annot_df["description"]== "Arousal", "description-fr"]="Eveil"

    # une obstruction des voies aériennes, qu’elle soit partielle (hypopnée) ou complète (apnée obstructive
    # SAOHS = Syndrome d’Apnées Hypopnées Obstructives du Sommeil 
    annot_df.loc[annot_df["description"]== "Hypopnea", "description-fr"]="Apnée hypopnée"
    annot_df.loc[annot_df["description"]== "Hypopnea", "Apnee"] = True
    annot_df.loc[annot_df["description"]== "Hypopnea", "code"]="SAOHS"

    annot_df.loc[annot_df["description"]== "Obstructive Apnea", "description-fr"]="Apnée obstructive"
    annot_df.loc[annot_df["description"]== "Obstructive Apnea", "Apnee"] = True
    annot_df.loc[annot_df["description"]== "Obstructive Apnea", "code"]="SAOHS"

    # incohérences respiratoires neurologiques centrales, c’est à dire des dysfonctionnements au niveau des centres nerveux, qui arrêtent alors de commander la respiration. 
    # SACS = Syndrome d’Apnées Centrales du Sommeil
    annot_df.loc[annot_df["description"]== "Central Apnea", "description-fr"]="Apnée centrale"
    annot_df.loc[annot_df["description"]== "Central Apnea", "Apnee"] = True
    annot_df.loc[annot_df["description"]== "Central Apnea", "code"]="SACS"

    annot_df["description"] = annot_df["description"].astype("category")
    annot_df["description_categ"] = annot_df["description"].cat.codes
    annot_df["description"] = annot_df["description"].astype("object")
    return annot_df


def resmed_get_group_df(annot_df, verbose=0):
    group = annot_df[annot_df['Apnee']==True].groupby(["day", 'code', "description_categ", "description"], as_index=True).agg({'Apnee':['count'], 
                         'duration':'sum'})
    group.columns = ['_'.join(col) for col in group.columns]
    group = group.reset_index()
    group = group.sort_values(by="day")
    group['date_new'] = pd.to_datetime(group['day'])
    group['date_new'] = group['date_new'].dt.strftime('%Y-%m-%d')
    return group


def resmed_merge_annotation_and_first(light_df, group, verbose=0):
    if verbose: print("Resmed > merge")
    df_merge = light_df.merge(group, left_on='Date-Real', right_on="day", how='left',indicator=True)
    if verbose: print("Resmed > merge > sorted rows")
    df_merge = df_merge.sort_values(by=["day"], ascending=False)
    if verbose: print("Resmed > merge > END")
    return df_merge

def resmed_postmerge_processing(resmed_merge, verbose=0):
    df_merge = resmed_merge.copy()
    if verbose: print("Resmed > postmerge > processing")

    if verbose: print("Resmed > postmerge > processing > date_new")
    df_merge['date_new'] = pd.to_datetime(df_merge['day'])
    df_merge['date_new'] = df_merge['date_new'].dt.strftime('%Y-%m-%d')

    if verbose: print("Resmed > postmerge > processing : Apnee_count")
    resmed_merge_completed = _complete_resmed_data(df_merge, col_to_proc="Apnee_count", suffix="_count", verbose=verbose)
    if verbose: print("Resmed > postmerge > processing : duration_sum")
    resmed_merge_completed = _complete_resmed_data(resmed_merge_completed, col_to_proc="duration_sum", suffix="_duration", verbose=verbose)
    
    if verbose: print("Resmed > postmerge > processing > remove and sorted columns")
    resmed_merge_completed = resmed_merge_completed[['day', 'date_new', 'Session duration in H', 'Duration',
       'Session duration sum in H', 
       'Central Apnea_count', 'Hypopnea_count', 'Obstructive Apnea_count', 
       'Central Apnea_duration', 'Hypopnea_duration', 'Obstructive Apnea_duration', 
       'MaskEvents', 'AHI', 'HI', 'AI', 'OAI', 'CAI', 'RIN', 'Crc16', 'MaskPress.Max', 'Leak.95', 'AmbHumidity.50',
       'Leak.50', 'MinVent.50',]]
    
    resmed_merge_completed = resmed_drop_duplicate(resmed_merge_completed, verbose=verbose)
    
    if verbose: print("Resmed > postmerge > processing > sorted rows")
    resmed_merge_completed = resmed_merge_completed.sort_values(by=["date_new"], ascending=False)
    if verbose: print("Resmed > postmerge > processing > END")
    return resmed_merge_completed


def resmed_drop_duplicate(df, verbose=0):
    df_merge = df.copy()
    if verbose: print("Resmed > drop duplicates")

    if verbose: print("Resmed > drop duplicates > remove duplicates row :", end="")
    if verbose: print("BEFORE:", df_merge.shape, end="")
    df_merge = df_merge.drop_duplicates(df_merge.columns, keep='first')
    if verbose: print("=>",df_merge.shape, end="")
    df_merge = df_merge[~df_merge["day"].isna()]
    if verbose: print("=>",df_merge.shape)
    
    if verbose: print("Resmed > drop duplicates > END")
    return df_merge


def resmed_complete_resmed_multi_data(resmed_merge, cols_to_proceed, verbose=0):

    resmed_merge_completed = resmed_merge.copy()

    for col_to_proc in cols_to_proceed:
        #concat with original data
        resmed_merge_completed = _complete_resmed_data(resmed_merge,col_to_proc,verbose )
    return resmed_merge_completed



from os.path import exists, getsize

def resmed_update_backup_files(backum_file_path, param_df, verbose=0):
    df_resmed = param_df.copy()
    if backum_file_path is not None and exists(backum_file_path) and getsize(backum_file_path) > 0 :
        f_df = None
        if verbose: print("Resmed > update backup > load backup data", backum_file_path, end="")
        f_df = pd.read_csv(backum_file_path, sep=",")
        if verbose: print("> ", f_df.shape, "loaded vs", df_resmed.shape, " proceeded")
        if verbose: print("Resmed > update backup > concat datas :", end="")
        data_light = pd.concat([df_resmed, f_df])
        if verbose: print(data_light.shape)
        
        data_light = resmed_drop_duplicate(data_light, verbose=verbose)
    else:
        if verbose: print("Resmed > update backup > backup file", backum_file_path, "NOT exist")
    if verbose: print("Resmed > update backup > end")
    return df_resmed


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Private function

def _get_annotations_by_date(datas_path, dataset_collecting_date, verbose=0):
    annotations_df = None
    
    for day in dataset_collecting_date:
        new_row = _get_annotation_for_date(datas_path,day, verbose=verbose)
        temp = None
        if new_row is not None and isinstance(new_row, pd.DataFrame):
            temp = new_row
        elif new_row is not None and len(new_row) > 0:
            aj_row = _ajust_dic_size(new_row)
            temp = pd.DataFrame.from_dict(aj_row)

        if annotations_df is None:
            annotations_df = temp
        elif temp is not None:
            annotations_df = pd.concat([annotations_df, temp],ignore_index=True, verify_integrity=False)

    return annotations_df

def _complete_resmed_data(resmed_merge, col_to_proc, suffix="", verbose=0):

    resmed_merge_completed = resmed_merge.copy()

    print(col_to_proc, end="")
    #concat with original data
    resmed_merge_completed = process_one_hot(resmed_merge_completed, col="description", verbose=verbose)

    unique_val = resmed_merge_completed["description"].unique().tolist()
    unique_val.remove(np.nan)

    for typ in unique_val:
        col_name = "description_"+typ
        print(":",col_name, end="")
        resmed_merge_completed.loc[resmed_merge_completed[col_name]>0,col_name] = resmed_merge_completed.loc[resmed_merge_completed[col_name]>0,col_to_proc]

    group = resmed_merge_completed.groupby(["day"], as_index=True).agg({'description_Central Apnea':['max'], 'description_Hypopnea':['max'], 'description_Obstructive Apnea':['max']})
    group.columns = [col[0] for col in group.columns]
    group = group.reset_index()
    group = group.sort_values(by="day")

    cols = list(group.columns)
    cols.remove('day')
    resmed_merge_completed = resmed_merge_completed.drop(columns=cols)
    resmed_merge_completed = resmed_merge_completed.merge(group, on='day', how='left')
    resmed_merge_completed = resmed_merge_completed.rename(columns={'description_Central Apnea':'Central Apnea'+suffix, 'description_Hypopnea':'Hypopnea'+suffix, 'description_Obstructive Apnea':'Obstructive Apnea'+suffix})
    print("    END")
    return resmed_merge_completed


def _get_edf_file_df(file_path, verbose=0):
    data = mne.io.read_raw_edf(file_path)
    # Réduction de l'échelle
    return data.to_data_frame() / 1000000


def _define_date(d):
    firt_val = 19018.0
    # 2022-01-26
    first_date = datetime(2022, 1, 26, 0, 0)
    nb_days = d - firt_val
    return first_date + timedelta(days=nb_days)


import mne

def _get_annotation_for_date(datas_path, day, verbose=0):
    date_dir_path = datas_path+day+"\\"
    res_df = None

    # récupération des fichiers du répertoire
    files = get_dir_files(date_dir_path, ".edf", verbose=verbose)
    if verbose > 1 :
        print(date_dir_path)

    row = None
    
    # on ne garde pas les dossiers vide
    if len(files) > 0:

        for f in files:
            annot = mne.read_annotations(date_dir_path + f)
            if isinstance(annot, mne.annotations.Annotations):
                
                annot_df = annot.to_data_frame()
                # On ajoute les colonnes dates, ...
                # Récupérer la date et l'heure du fichier
                date_time = f[0:-8]
                annot_df["date_time"] = datetime.strptime(date_time, '%Y%m%d_%H%M%S')                
                annot_df["day"] = datetime.strptime(day, '%Y%m%d')
                annot_df["type"] = f[-7:-4]
                                
                if res_df is None:
                    res_df = annot_df
                else:
                    res_df = pd.concat([res_df, annot_df],ignore_index=True, verify_integrity=False)                
        if verbose>1:
            print(row)

    return res_df


def _ajust_dic_size(row, verbose=0):
    if row is not None:
        
        max_col = 0
        # Rechercher le nombre max de valeurs
        for key, vals in max_col.items():
            if len(vals)> max_col:
                max_col = len(vals)

        # Ajout des NAN pour que les tableaux soient tous de la même taille        
        for key in row.keys():
            val = np.nan
            if 'day' in key or len(row[key]) == 1:
                val = row[key][0]
            
            for i in range (len(row[key]), max_col):
                row[key].append(val)
    return row