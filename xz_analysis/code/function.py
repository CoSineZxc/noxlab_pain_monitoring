import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import pearsonr
import pandas as pd

class pain_rating_D1D2:
    def __init__(self,PID):
        self.PID=PID
        self.D1_rating=np.nan
        self.D1_time=np.nan
        self.D1_prediction=np.nan
        self.D1_confidence=np.nan
        self.D2_rating=np.nan
        self.D2_time=np.nan
        self.D2_prediction=np.nan
        self.D2_confidence=np.nan
    def insert_rating_time_D1(self,rating,time):
        self.D1_rating=rating
        self.D1_time=time
    def insert_rating_time_D2(self,rating,time):
        self.D2_rating=rating
        self.D2_time=time
    def update_prediction_confidence_D1(self,pred,conf):
        self.D1_prediction=pred
        self.D1_confidence=conf
    def update_prediction_confidence_D2(self,pred,conf):
        self.D2_prediction=pred
        self.D2_confidence=conf
    def glue_data(self,Day):
        rating_1D=[]
        if Day==1:
            rating_2D=self.D1_rating
        else:
            rating_2D=self.D2_rating
        for trial in rating_2D:
            rating_1D+=trial.tolist()
        # return self.D1_rating[1]
        return rating_1D
    def mean_rating(self,Day):
        rating_1D=self.glue_data(Day)
        return np.mean(rating_1D)
    def std(self,Day):
        rating_1D=self.glue_data(Day)
        return np.std(rating_1D)
    def cv(self,Day):
        return self.std(Day)/self.mean_rating(Day)
    def fanofactor(self,Day):
        return (self.std(Day)*self.std(Day))/self.mean_rating(Day)
    def IQR(self, Day):
        rating_1D=self.glue_data(Day)
        q1 = np.percentile(rating_1D, 25)
        q3 = np.percentile(rating_1D, 75)
        iqr = q3 - q1
        return iqr
    def RMSE(self):
        rating_1D=self.glue_data(2)
        rating_1D=np.array(rating_1D)
        squared_diff = (rating_1D - self.D1_prediction) ** 2
        mse = np.mean(squared_diff)
        rmse = np.sqrt(mse)
        return rmse
    def fft_data(self, Day, Fre):
        rating=self.glue_data(Day)
        time=np.arange(0,len(rating)*(1/Fre),1/Fre)
        fft_weights = np.fft.fft(rating)
        N = len(fft_weights)
        n = np.arange(N)
        T = time[-1]-time[0]
        fft_freq = n/T
        return fft_weights, fft_freq
    # power spectrum
    def power_spectrum(self, Day, Fre,plot=False):
        rating=self.glue_data(Day)
        # rating_downsample=signal.resample(rating, int(len(rating)/2))
        frequencies, psd = welch(rating, Fre)
        # f, S = welch(rating_downsample, Fre/2)
        Fre_max=frequencies[psd.argmax()] #max Fre
        Amp_max=max(psd)
        if plot==True:
            plt.semilogy(frequencies,psd)
            plt.xlim(0,2.5)
            plt.show()
        return Fre_max,Amp_max
    
def p_value_correction_rectangle_fdr(p_value_matrix,standard_alpha):
    p_values_list = p_value_matrix.flatten()
    fdr_result=fdrcorrection(p_values_list,standard_alpha)
    rejected_indices = [index for index,value in enumerate(fdr_result[0]) if value == True]
    # print(rejected_indices)
    # Create a matrix to store the corrected significance levels
    corrected_matrix = np.ones_like(p_value_matrix)
    # print(corrected_matrix)
    # Populate the corrected significance levels in the matrix
    # tri_index=np.triu_indices_from(p_value_matrix, k=1)
    num_rows, num_cols = p_value_matrix.shape
    x_indices, y_indices = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
    tri_index = np.vstack((y_indices.flatten(),x_indices.flatten()))
    # print(tri_index)
    for index in rejected_indices:
        # print("index: "+str(index))
        row=tri_index[0][index]
        col = tri_index[1][index]
        # print("row: "+str(row)+", col: "+str(col))
        corrected_matrix[row, col] = p_values_list[index]

    # # Fill the lower triangular part of the matrix symmetrically
    # corrected_matrix = corrected_matrix 
    return corrected_matrix

def p_value_correction_square_fdr(p_value_matrix,standard_alpha):
    p_values_list = p_value_matrix[np.triu_indices_from(p_value_matrix, k=1)]
    fdr_result=fdrcorrection(p_values_list,standard_alpha)
    rejected_indices = [index for index,value in enumerate(fdr_result[0]) if value == True]
    # print("rejected_indices: "+str(rejected_indices))
    # print(p_values)
    # print(correlation_matrix)
    # Create a matrix to store the corrected significance levels
    corrected_matrix = np.ones_like(p_value_matrix)
    # print(corrected_matrix)
    # Populate the corrected significance levels in the matrix
    for index in rejected_indices:
        # print("index: "+str(index))
        tri_index=np.triu_indices_from(p_value_matrix, k=1)
        row=tri_index[0][index]
        col = tri_index[1][index]
        # print("row: "+str(row)+", col: "+str(col))
        corrected_matrix[row, col] = p_values_list[index]
    # Fill the lower triangular part of the matrix symmetrically
    corrected_matrix = corrected_matrix + corrected_matrix.T-np.ones_like(p_value_matrix)
    return corrected_matrix

def datadiv(data,div):
    data_np=np.array(data)/div
    data=data_np.tolist()
    return data

class Carl_pain_rating:
    def __init__(self,PID):
        self.PID=PID
        self.rating=np.nan
        self.time=np.nan
        self.Fre=np.nan
    def insert_rating_time_D1(self,rating,Fre):
        self.rating=rating
        self.Fre=Fre
        self.time=np.arange(0,len(rating)*(1/Fre),1/Fre).tolist()
    def mean_rating(self):
        return np.mean(self.rating)
    def std(self):
        return np.std(self.rating)
    def cv(self):
        return self.std()/self.mean_rating()
    def fanofactor(self):
        return (self.std()*self.std())/self.mean_rating()
    def IQR(self):
        q1 = np.percentile(self.rating, 25)
        q3 = np.percentile(self.rating, 75)
        iqr = q3 - q1
        return iqr
    def fft_data(self):
        fft_weights = np.fft.fft(self.rating)
        N = len(fft_weights)
        n = np.arange(N)
        T = self.time[-1]-self.time[0]
        fft_freq = n/T
        return fft_weights, fft_freq
    # power spectrum
    def power_spectrum(self,plot=False):
        frequencies, psd = welch(self.rating, self.Fre)
        f_max=frequencies[psd.argmax()]
        if plot==True:
            plt.semilogy(frequencies,psd)
            plt.xlim(0,2.5)
            plt.show()
        return f_max
    
class pain_pred_acc_clinical:
    def __init__(self,PID):
        self.PID=PID
        self.mean1=np.nan
        self.std1=np.nan
        self.cv1=np.nan
        self.IQR1=np.nan
        self.fano1=np.nan
        self.psd1=np.nan
        self.mean2=np.nan
        self.std2=np.nan
        self.cv2=np.nan
        self.IQR2=np.nan
        self.fano2=np.nan
        self.psd2=np.nan
        self.prediction=np.nan
        self.confidence=np.nan
        self.acc=np.nan
        self.nstate_D1=np.nan
        self.hmmscore_D1=np.nan
        self.nstate_D2=np.nan
        self.hmmscore_D2=np.nan
        self.msk=np.nan
        self.BPI_severity=np.nan
        self.pcs_total=np.nan
        self.pcs_rumination=np.nan
        self.pcs_magnification=np.nan
    def update(self,pain_rating,nstate1,hmmscore1,nstate2,hmmscore2,msk,BPI_S,PCS_T,PCS_R,PCS_M):
        self.mean1=pain_rating.mean_rating(1)
        self.mean2=pain_rating.mean_rating(2)
        self.std1=pain_rating.std(1)
        self.std2=pain_rating.std(2)
        self.cv1=pain_rating.cv(1)
        self.cv2=pain_rating.cv(2)
        self.IQR1=pain_rating.IQR(1)
        self.IQR2=pain_rating.IQR(2)
        self.fano1=pain_rating.fanofactor(1)
        self.fano2=pain_rating.fanofactor(2)
        self.psd1=pain_rating.power_spectrum(1,40)[1]
        self.psd2=pain_rating.power_spectrum(2,40)[1]
        self.prediction=pain_rating.D1_prediction
        self.confidence=pain_rating.D1_confidence
        self.acc=1-pain_rating.RMSE()
        self.nstate_D1=nstate1
        self.hmmscore_D1=hmmscore1
        self.nstate_D2=nstate2
        self.hmmscore_D2=hmmscore2
        if not np.isnan(msk):
            self.msk=msk
        if not np.isnan(BPI_S):
            self.BPI_severity=BPI_S
        if not np.isnan(PCS_T):
            self.pcs_total=PCS_T
        if not np.isnan(PCS_R):
            self.pcs_rumination=PCS_R
        if not np.isnan(PCS_M):
            self.pcs_magnification=PCS_M
    def check_isnan(self):
        if np.isnan(self.mean1) or np.isnan(self.std1) or np.isnan(self.cv1) or np.isnan(self.IQR1) or np.isnan(self.fano1) or np.isnan(self.psd1) or\
                np.isnan(self.mean2) or np.isnan(self.std2) or np.isnan(self.cv2) or np.isnan(self.IQR2) or np.isnan(self.fano2) or np.isnan(self.psd2) or\
                np.isnan(self.prediction) or np.isnan(self.confidence) or np.isnan(self.acc) or\
                np.isnan(self.nstate_D1) or np.isnan(self.nstate_D2) or\
                np.isnan(self.msk) or np.isnan(self.BPI_severity) or\
                np.isnan(self.pcs_total) or np.isnan(self.pcs_rumination) or np.isnan(self.pcs_magnification):
            print(self.PID+" invalid")
            return True
        else:
            return False
        
class pain_pred_acc_conf:
    def __init__(self,PID):
        self.PID=PID
        self.mean2=np.nan
        self.std2=np.nan
        self.cv2=np.nan
        self.IQR2=np.nan
        self.fano2=np.nan
        self.psd2=np.nan
        self.prediction=np.nan
        self.acc=np.nan
        self.confidence=np.nan
    def update(self,pain_rating):
        self.mean2=pain_rating.mean_rating(2)
        self.std2=pain_rating.std(2)
        self.cv2=pain_rating.cv(2)
        self.IQR2=pain_rating.IQR(2)
        self.fano2=pain_rating.fanofactor(2)
        self.psd2=pain_rating.power_spectrum(2,40)[1]
        self.prediction=pain_rating.D1_prediction
        self.confidence=pain_rating.D1_confidence
        self.acc=10-pain_rating.RMSE()
    def check_isnan(self):
        if np.isnan(self.mean2) or np.isnan(self.std2) or np.isnan(self.cv2) or np.isnan(self.IQR2) or np.isnan(self.fano2) or np.isnan(self.psd2) or\
                np.isnan(self.prediction) or np.isnan(self.confidence) or np.isnan(self.acc):
            print(self.PID+" invalid")
            return True
        else:
            return False
        
class pain_rating_pred_hmm:
    def __init__(self,PID):
        self.PID=PID
        self.mean1=np.nan
        self.std1=np.nan
        self.cv1=np.nan
        self.IQR1=np.nan
        self.fano1=np.nan
        self.mean2=np.nan
        self.std2=np.nan
        self.cv2=np.nan
        self.IQR2=np.nan
        self.fano2=np.nan
        self.prediction=np.nan
        self.confidence=np.nan
        self.acc=np.nan
        self.nstate_D1=np.nan
        self.hmmscore_D1=np.nan
        self.nstate_D2=np.nan
        self.hmmscore_D2=np.nan
    def update(self,pain_rating,nstate1,hmmscore1,nstate2,hmmscore2):
        self.mean1=pain_rating.mean_rating(1)
        self.mean2=pain_rating.mean_rating(2)
        self.std1=pain_rating.std(1)
        self.std2=pain_rating.std(2)
        self.cv1=pain_rating.cv(1)
        self.cv2=pain_rating.cv(2)
        self.IQR1=pain_rating.IQR(1)
        self.IQR2=pain_rating.IQR(2)
        self.fano1=pain_rating.fanofactor(1)
        self.fano2=pain_rating.fanofactor(2)
        self.prediction=pain_rating.D1_prediction
        self.confidence=pain_rating.D1_confidence
        self.acc=10-pain_rating.RMSE()
        self.nstate_D1=nstate1
        self.hmmscore_D1=hmmscore1
        self.nstate_D2=nstate2
        self.hmmscore_D2=hmmscore2
    def check_isnan(self):
        if np.isnan(self.mean1) or np.isnan(self.std1) or np.isnan(self.cv1) or np.isnan(self.IQR1) or np.isnan(self.fano1) or\
                np.isnan(self.mean2) or np.isnan(self.std2) or np.isnan(self.cv2) or np.isnan(self.IQR2) or np.isnan(self.fano2) or\
                np.isnan(self.prediction) or np.isnan(self.confidence) or np.isnan(self.acc) or\
                np.isnan(self.nstate_D1) or np.isnan(self.nstate_D2):
            print(self.PID+" invalid")
            return True
        else:
            return False
        
class part_for_paper:
    def __init__(self,PID):
        self.PID=PID
        self.age=np.nan
        self.gender=np.nan
        self.MSK=np.nan
    def update(self,age,gender,MSK):
        if type(age)==str:
            self.age=int(age)
        if type(gender)==str:
            self.gender=gender
        if type(MSK)==float:
            self.MSK=int(MSK)

def cal_corr_pvalues(df):
    cormat = df.corr()
    pvalues = df.corr(method=lambda x, y: pearsonr(x, y)[1])
    return cormat,pvalues

def cut_corrected(cormat,pvalues,iloc_x_1=0,iloc_x_2=0,iloc_y_1=0,iloc_y_2=0,print_value=True):
    cormat_part=cormat.iloc[iloc_x_1:iloc_x_2,iloc_y_1:iloc_y_2]
    pvalues_part=pvalues.iloc[iloc_x_1:iloc_x_2,iloc_y_1:iloc_y_2]
    corrected_pvalues_part=p_value_correction_rectangle_fdr(pvalues_part.to_numpy(),0.05)
    corrected_pvalue_part_df = pd.DataFrame(data = corrected_pvalues_part,\
        index = pvalues_part.index,\
            columns = pvalues_part.columns)
    if print_value==True:
        print(cormat_part)
        print(corrected_pvalue_part_df)
    return cormat_part,corrected_pvalue_part_df