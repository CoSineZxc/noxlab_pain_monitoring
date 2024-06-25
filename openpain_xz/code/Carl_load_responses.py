from bids import BIDSLayout
import json
import pandas as pd
import os

import gzip
import matplotlib.pyplot as plt
import numpy as np


def norm(x):
  return (x - np.mean(x))

def readCsv(file_path):
  path = "Carl_preprocessed_responses/"
  return np.array(pd.read_csv(os.path.join(path,file_path), delimiter='\t', header=None))

def get_all_visits_for_one_subject(participant):
    """
    returns a list of ints representing the visits that the participant was at, using the sessions.tsv file
    """
    session_file = f"/rds/project/rds-3IOyKgCQu4I/sbp/openpain.org/subacute_longitudinal_study/{participant}/{participant}_sessions.tsv"
    f = pd.read_csv(session_file, sep='\t')
    return [i[-1] for i in list(f['session_id'].loc[f['session_id'].str.match('visit*')])]

def get_all_responses_for_one_subject(subject, task_type,downsampled=True, verbose = True):
    """Gets all the pain responses across sessions and runs for one subject."""
    responses = []
    key = []
    
    visits = get_all_visits_for_one_subject(subject)
    if verbose:
        print(subject, visits)
    
    for visit in visits:
        for run in [1,2]:
            try:
                response = load_single_subject_response(subject,visit,run,plot=0,task_type=task_type, display = False,downsample=downsampled)
                responses.append(response)
                key.append(f'visit{visit}, run_0{run}')
            except Exception as err:
                if verbose:
                    print(err)
                continue
            
            
    return np.array(responses, dtype='object'),key 

def plot_all_responses_for_one_subject(subject, task_type,downsampled = False, fs = 14.40):

    fig, ax = plt.subplots(figsize=(15,6))

    responses, key = get_all_responses_for_one_subject(subject, task_type, downsampled = downsampled)

    for response in responses:
        ax.plot([i/fs for i in range(len(response))], response) # downsizing responses
    plt.ylim(0, 110)
    plt.xlabel('seconds')
    plt.legend(key)
    plt.title(f"Pain ratings across runs for subject {subject} on the {task_type} task, downsampled = {downsampled}")
    return responses,key
      
def load_single_subject_response(subject: str, visit: int, run: int, plot: int = 0,task_type = 'sp', display = True,downsample=True):
    """
    Loads a single subject response from a task within a run and visit, and optionally plots it
    
    Parameters
    ----------
    participant_id: str
        Examples: "sub-001", "sub-002", ... "sub-122".
        Range: ["001", "122"]
    
    visit: int
        Examples: 1, 2, .. 5
        Range: [1, 5] but for some participants it's [1, 4]
    
    run: int
        Examples: 1, 2
        Range: [1, 2]
    
    plot: Bool
        Plots data if True, else does not
       
    task_type: str
        Examples: "sp","sv","mv"
        defaults to "sp"
    
    Returns
    -------
    downsampled response : numpy.ndarray
    
    else throws an error stating the reponse that could not be found
    """
    participants_df = pd.read_csv('openpain.org/subacute_longitudinal_study/participants.tsv', sep='\t')
    
    if display:
        print("loading the data of: ")
        print(participants_df.loc[participants_df['participant_id'] == subject, ["group", "race", "gender", "age", "origin"]])
    session = "visit" + str(visit)

    if task_type == 'sp':
        resp_file = f"openpain.org/subacute_longitudinal_study/{subject}/ses-visit{visit}/func/{subject}_ses-visit{visit}_task-sp_run-0{run}_resp.tsv.gz"
    else:
        resp_file = f"openpain.org/subacute_longitudinal_study/{subject}/ses-visit{visit}/func/{subject}_ses-visit{visit}_task-{task_type}_resp.tsv.gz"
        
    try:
        f = gzip.open(resp_file, 'rb')
    except:
        err = f"resp not available, subject {subject}, visit {visit}, run {run}"
        raise ValueError(err)

    response = f.read().decode("utf-8").split("\n")
    to_delete = []
    for i in range(0, len(response)):
        try:
            response[i] = float(response[i])
        except:
            to_delete.append(i)

    for i in to_delete:
        del response[i]

    if (len(response) != 244) and downsample:
        response = response[:8784:36] 
    if display:
        print("len of responses: ", len(response))
    if plot:
        fig, ax = plt.subplots(figsize=(16,6))
        plt.title(f"{'Downsampled' if downsample else ''} Responses")
        ax.plot(response)
    return np.array(response)