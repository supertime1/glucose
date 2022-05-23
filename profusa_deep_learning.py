"""
This document includes functions to estimate blood glucose via a deep learning approach
"""
import mlflow
import mlflow.sklearn
import os
import glob
import numpy as np
from datetime import datetime
import scipy
import scipy.signal as signal
from scipy.stats import uniform, randint
import pickle
from profusa_glucose import *
from profusa_const import GCol
from profusa_glucose_ref_glu import *
from study_parameters import Clinical_Study
import profusa_glucose_util as pgutil
import profusa_glucose_analysis as pga
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV, GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class DLConst:
    """
    Background ML project constants definitions
    """
    # Feature engineering steps (i.e. all steps to extract input ML features)
    FEATURE_ENGINEERING_STEPS = {'acc_trans': True,
                                 'led_pd_trans': False,
                                 'led_pd_all': True,
                                 'led_pd_var': False,
                                 'opt_property': False,
                                 'ambient_light': True,
                                 'individual_pmt': True,
                                 'temperature_all': True
                                 }
    # pgdf columns names that will be used as labels for bkg ML model (aka. SIGNAL_LED_GLU_PMT / REF_AC_FFT)
    LABEL_COL_NAMES = 'spline_BG'
    # mlflow experiment name
    EXPERIMENT_NAME = 'background'
    
    PD_WAVELENGTH_DISTANCES = {
        610: {
            1: ['led1_pd6_ac_(mV)', 'led2_pd6_ac_(mV)', 'led3_pd4_ac_(mV)', 'led4_pd4_ac_(mV)'],
            2: ['led1_pd1_ac_(mV)', 'led2_pd3_ac_(mV)', 'led3_pd5_ac_(mV)', 'led4_pd2_ac_(mV)'],
            3: ['led1_pd5_ac_(mV)', 'led2_pd2_ac_(mV)', 'led3_pd1_ac_(mV)', 'led4_pd3_ac_(mV)'],
            4: ['led1_pd4_ac_(mV)', 'led2_pd4_ac_(mV)', 'led3_pd6_ac_(mV)', 'led4_pd6_ac_(mV)'],
            5: ['led1_pd3_ac_(mV)', 'led2_pd1_ac_(mV)', 'led3_pd2_ac_(mV)', 'led4_pd5_ac_(mV)']},
        740: {
            1: ['led5A_pd2_ac_(mV)', 'led6A_pd3_ac_(mV)', 'led7A_pd3_ac_(mV)', 'led8A_pd2_ac_(mV)'],
            2: ['led5A_pd5_ac_(mV)', 'led6A_pd6_ac_(mV)', 'led7A_pd1_ac_(mV)', 'led8A_pd4_ac_(mV)'],
            3: ['led5A_pd1_ac_(mV)', 'led6A_pd4_ac_(mV)', 'led7A_pd5_ac_(mV)', 'led8A_pd6_ac_(mV)'],
            4: ['led5A_pd3_ac_(mV)', 'led6A_pd2_ac_(mV)', 'led7A_pd2_ac_(mV)', 'led8A_pd3_ac_(mV)'],
            5: ['led5A_pd4_ac_(mV)', 'led6A_pd1_ac_(mV)', 'led7A_pd6_ac_(mV)', 'led8A_pd5_ac_(mV)']},
        810: {
            1: ['led9_pd5_ac_(mV)', 'led10_pd5_ac_(mV)', 'led11_pd1_ac_(mV)', 'led12_pd1_ac_(mV)'],
            2: ['led9_pd2_ac_(mV)', 'led10_pd4_ac_(mV)', 'led11_pd3_ac_(mV)', 'led12_pd6_ac_(mV)'],
            3: ['led9_pd3_ac_(mV)', 'led10_pd6_ac_(mV)', 'led11_pd2_ac_(mV)', 'led12_pd4_ac_(mV)'],
            4: ['led9_pd1_ac_(mV)', 'led10_pd1_ac_(mV)', 'led11_pd5_ac_(mV)', 'led12_pd5_ac_(mV)'],
            5: ['led9_pd4_ac_(mV)', 'led10_pd2_ac_(mV)', 'led11_pd6_ac_(mV)', 'led12_pd3_ac_(mV)']}
        }

    # delay bg measurement in skin comparing to blood vessels
    BG_TIME_DELAY = 7.5 * 60 #seconds
    
    # define maximum time span allowance in seconds in one training window, in case remove bad sample remove too many points
    MAX_TIME_SPAN_ALLOWANCE = 22 * 60  # seconds

    
class DataImporter():
    """
    DataImporter class defines functions to import data to be processed by Pipeline class
    """
    @staticmethod
    def find_background_from_local(site_name='Site 59 - In Vitro'):
        """
        Find background files from Site 59 that are collected from volunteers
        """
        # find all experiment paths
        clinical_data_root = pgutil.clinical_data_root()
        folder_name = '59-background_ml'
        subfolder_name_lst = ['adhesive_validation',
                              'long_measurements', 
                              'short_skin_meas',
                              'background']
        experiment_paraent_paths = glob.glob(os.path.join(clinical_data_root, site_name, folder_name))

        adhesive_exp_paths = glob.glob(os.path.join(experiment_paraent_paths[0], subfolder_name_lst[0], '*'))
        long_exp_paths = glob.glob(os.path.join(experiment_paraent_paths[0], subfolder_name_lst[1], '*'))
        short_exp_paths = glob.glob(os.path.join(experiment_paraent_paths[0], subfolder_name_lst[2], '*'))
        site59_bkg_2021_exp_paths = glob.glob(os.path.join(experiment_paraent_paths[0], subfolder_name_lst[3], '*'))

        print(f'There {len(adhesive_exp_paths)} background files in adhesive_exp_paths.')
        print(f'There {len(long_exp_paths)} background files in long_exp_paths.')
        print(f'There {len(short_exp_paths)} background files in short_exp_paths.')
        print(f'There {len(site59_bkg_2021_exp_paths)} background files in site59_bkg_2021_exp_paths.')

        return adhesive_exp_paths, long_exp_paths, short_exp_paths, site59_bkg_2021_exp_paths

    @staticmethod
    def find_background_from_clinical(site_filter, day_post_injection_filter, bkg_std_filter):
        """
        Find background files from clinical studies from different sites, by applying different filters
        """
        # find experimetn quality file
        experiment_quality_file_path = os.path.join(pgutil.clinical_data_root(),
                                                    "experiment_quality.csv")
        try:
            experiment_quality = pd.read_csv(experiment_quality_file_path)
        except:
            print(
                f"No experiment quality file found. {experiment_quality_file_path}")
            raise
        print(len(experiment_quality), "experiments")

        # filter experiment by site
        if site_filter != '':
            experiment_quality = experiment_quality[experiment_quality['site'] == int(
                site_filter)]
        else:
            experiment_quality = experiment_quality[(
                experiment_quality['site'] >= 1)]
        print(
            f'After site filter (site={site_filter}) experiments={len(experiment_quality.index)}')

        # filter experiment by days post sensor injection
        experiment_quality = experiment_quality[experiment_quality['day_post_injection']
                                                >= day_post_injection_filter]
        print(
            f'After day post injection filter (day>={day_post_injection_filter}) experiments={len(experiment_quality.index)}')

        # filter experiment by background std
        experiment_quality = experiment_quality[experiment_quality['background_std_calibrated'] < bkg_std_filter]

        bkg_file_lst = []
        for experiment_path in pgutil.get_all_clinical_experiments():
            experiment_quality_rows = experiment_quality[experiment_quality['experiment_file'] == os.path.basename(
                experiment_path)]
            if len(experiment_quality_rows) > 0:
                if len(pgutil.find_background(experiment_path)) > 0:
                    bkg_file_lst.append(
                        pgutil.find_background(experiment_path))
        num_experiment = len(bkg_file_lst)
        bkg_file_lst = [i for sublist in bkg_file_lst for i in sublist]
        print(
            f'There are {len(bkg_file_lst)} background files from {num_experiment} experiments')

        return bkg_file_lst

    @staticmethod
    def find_background_files_pig():
        """
        Find background skin measurement files from pig experiment
        """
        experiment_paths = pgutil.find_experiments(site='00', subject='', date_part='', sensor='', device='')
        pig_bkg_files_lst = [_ for _ in experiment_paths if 'X' in os.path.basename(_).split('_')[1]]
        print(f'There are {len(pig_bkg_files_lst)} pig background files.')
        return pig_bkg_files_lst


class FeatureEngineer():
    """
    This class defines functions to do feature engineering
    """
    def __init__(self, experiment: Experiment):
        self.exp = experiment
        # patient temperture is included in input features by default
        self.feature_col_names = []

    def acceleration_transform(self):
        """
        Transform accelerations: acc_trans = sqrt(x^2 + y^2 + z^2)
        """
        acc_x = self.exp.pgdf[GCol.ACCELERATION_X].values
        acc_y = self.exp.pgdf[GCol.ACCELERATION_Y].values
        acc_z = self.exp.pgdf[GCol.ACCELERATION_Z].values
        self.exp.pgdf['acc_trans'] = np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))
        GCol.ACCELERATION_TRANS = 'acc_trans'
        self.feature_col_names.append(GCol.ACCELERATION_TRANS)
        return self.exp

    def signal_led_pd_transform(self):
        """
        Transform led_pd signals by taking the average value of each wavelength
        """
        for key, value in DLConst.PD_WAVELENGTH_DISTANCES.items():
            for sub_k, sub_v in value.items():
                new_col_name = str(key) + '_' + str(sub_k) + '_mean'
                self.exp.pgdf[new_col_name] = np.mean(self.exp.pgdf[sub_v].values, axis=1)
                self.feature_col_names.append(new_col_name)
        return self.exp

    def signal_led_pd_all_transform(self):
        """
        All all 72 led wavelength - distance combination
        """
        for _, value in DLConst.PD_WAVELENGTH_DISTANCES.items():
            for _, sub_v in value.items():
                for col_name in sub_v:
                    self.feature_col_names.append(col_name)
        return self.exp

    def optical_property_transform(self):
        """
        Generate optical properties as ML inputs features
        """
        self.exp.compute_optical()
        for opt_prop in GCol.OPTICAL_PROPERTIES:
            self.feature_col_names.append(opt_prop)
        return self.exp

    def signal_led_pd_var_transform(self):
        """
        Generate variations of LED pd signals as ML inputs features
        """
        # GCol.SIGNAL_LED_PD_VAR = []
        for key, value in DLConst.PD_WAVELENGTH_DISTANCES.items():
            for sub_k, sub_v in value.items():
                new_col_name = str(key) + '_' + str(sub_k) + '_var'
                self.exp.pgdf[new_col_name] = np.var(self.exp.pgdf[sub_v].values, axis=1)
                self.feature_col_names.append(new_col_name)
        return self.exp

    def temperature_all_transform(self):
        """ add all temperature cols """
        for col_name in GCol.TEMPERATURES:
            self.feature_col_names.append(col_name)
        return self.exp    
    
    def ambient_light_transform(self):
        """ add led0_pmt_ac_(mV) """
        self.feature_col_names.append(GCol.DARK_PMT)
        return self.exp
    
    def individual_pmt_transform(self):
        """ add individual pmt signals """
        for col_name in GCol.SIGNAL_LED_GLU_PMT + GCol.SIGNAL_LED_REF_PMT:
            self.feature_col_names.append(col_name)
        return self.exp


class Pipeline():
    """
    Pipeline class defines all steps to process data from a list of data file directories
    """

    def __init__(self, experiment_paths, num_of_inputs, stride, exp_type='h') -> None:
        # experiment_paths_list is a list of experiment path list
        self.experiment_paths = list(experiment_paths)
        self.num_of_inputs = num_of_inputs
        self.stride = stride
        # Track unqualified experiment when the time interval between consecutive samples are different
        self.unqualified_exp_lst = []
        # record the basename of experiment file
        self.data_file_names = [os.path.basename(_) for _ in experiment_paths]
        self.feature_col_names = []
        self.exp_type = exp_type

    def input_pipeline(self):
        """
        Process experiment paths from different directories, and store the data and labels into a array
        of lists, that each list is an experiment contains data and labels that corresponding to that experiment(aka. list)
        """
        all_feature_lst = []
        all_label_lst = []
        # shuffle the experiment path list to help cross validation splits train-val ratio better
        np.random.seed(7)
        np.random.shuffle(self.experiment_paths)

        total_exp = len(self.experiment_paths)
        for idx, experiment_path in enumerate(self.experiment_paths):
            if idx % 10 == 0:    
                print(f'processing {round((idx + 1) / total_exp * 100, 2)}% experiment....')
            # load experiment
            try:
                exp = Experiment(experiment_path, prep=self.exp_type)
            except:
                print('Warning: Failed to process experiment!')
                continue
            # load blood glucose file
            try:
                reference_files = pgutil.find_blood_glucose_reference(experiment_path)
                tz = Clinical_Study.SITE_TIMEZONES[exp.subject[0:2]]
                blood_glucose = Reference_Glucose(reference_files[0], timezone_region=tz)
            except Exception as exc:
                print(f'Failed to load reference blood glucose data {exc}')
                continue
            # shift blood glucose time to compensate the skin measurement delay
            blood_glucose.df['Timestamp(UTC[s])'] += DLConst.BG_TIME_DELAY
            exp.pgdf = add_glu_spline_df_to_dataframe(exp.pgdf, blood_glucose.df)
            assert DLConst.LABEL_COL_NAMES in exp.pgdf.columns
            # prepocess the experiment to remove bad samples, HWCal and do feature transform
            processed_exp = self.preprocess_experiment(exp)
            assert DLConst.LABEL_COL_NAMES in processed_exp.pgdf.columns

            if not processed_exp:
                print('experiment is too short after removing bad sample')
                continue            
            # extract features and labels
            feature_lst, label_lst = self.feature_label_extraction(processed_exp)
            all_feature_lst += feature_lst
            all_label_lst += label_lst

        all_features = np.array(all_feature_lst)
        all_labels = np.array(all_label_lst)

        # print(
        #     f"Data is from {len(all_features)} experiments, and the shape for each experiment is : {all_features[0][0].shape}")
        # print(
        #     f"Label shape is from {len(all_labels)} experiments, and the shape for each experiment is: {all_labels[0][0].shape}")

        return all_features, all_labels

    def preprocess_experiment(self, experiment: Experiment):
        """
        Perform hardware calibration, filter_raw_signals and other preprocessing steps
        if necessary (e.g. feature transformation)
        """
        try:
            experiment.remove_bad_samples(remove_ambient_light=True, remove_low_reference=False,
                                          remove_large_motion=False, remove_rapid_temperature_change=False)
        except ValueError:
            print('Not enough data to process after removing bad samples!')
            return None
        
        experiment.hardware_calibrate()
        # experiment.pgdf = remove_corrupt_pd_samples(experiment.pgdf)
        feature_eng = FeatureEngineer(experiment)
        for step_name, value in DLConst.FEATURE_ENGINEERING_STEPS.items():
            if step_name == 'acc_trans' and value is True:
                processed_exp = feature_eng.acceleration_transform()
            if step_name == 'led_pd_trans' and value is True:
                processed_exp = feature_eng.signal_led_pd_transform()
            if step_name == 'led_pd_all' and value is True:
                processed_exp = feature_eng.signal_led_pd_all_transform()
            if step_name == 'led_pd_var' and value is True:
                processed_exp = feature_eng.signal_led_pd_var_transform()
            if step_name == 'opt_property' and value is True:
                processed_exp = feature_eng.optical_property_transform()
            if step_name == 'ambient_light' and value is True:
                processed_exp = feature_eng.ambient_light_transform()
            if step_name == 'individual_pmt' and value is True:
                processed_exp = feature_eng.individual_pmt_transform()
            if step_name == 'temperature_all' and value is True:
                processed_exp = feature_eng.temperature_all_transform()
            else:
                processed_exp = experiment
        self.feature_col_names = feature_eng.feature_col_names
        return processed_exp

    def feature_label_extraction(self, experiment: Experiment):
        """
        Extract the background and features arrays from each experiment, start with second point of the file
        to avoid uncertainty of the first point
        """
        # extract feature columns from pgdf
        timestamp_col = experiment.pgdf[GCol.TIMESTAMP_UTC].values
        feature_cols = experiment.pgdf[self.feature_col_names].values
        label_cols = experiment.pgdf[DLConst.LABEL_COL_NAMES].values
        # initiate feature lst and label lst
        feature_lst = []
        label_lst = []
        # use a sliding window with a stride to extract input features for ML model
        for i in range(0, len(feature_cols) - self.num_of_inputs, self.stride):
            if timestamp_col[i + self.num_of_inputs] - timestamp_col[i] > DLConst.MAX_TIME_SPAN_ALLOWANCE:
                continue
            feature_lst.append(feature_cols[i:i + self.num_of_inputs])
            label_lst.append(label_cols[i:i + self.num_of_inputs])
        return feature_lst, label_lst

    @staticmethod
    def train_data_split(experiment_paths, train_ratio, seed=7):
        """
        Train - test data split based on train_ratio
        """
        # to empty test_data_file_names
        np.random.seed(seed)
        np.random.shuffle(experiment_paths)
        # calculate train, val, test number of files
        total_num_files = len(experiment_paths)
        num_test_files = int(np.ceil((1 - train_ratio) * total_num_files))
        num_train_files = total_num_files - num_test_files

        test_exp_paths = experiment_paths[:num_test_files]
        train_exp_paths = experiment_paths[num_test_files:]

        # self.test_data_file_names += list(test_exp_paths)
        print(f'Num of train files {num_train_files}')
        print(f'Num of test files {num_test_files}')

        return train_exp_paths, test_exp_paths

    @staticmethod
    def save_train_test_dataset(train_data, train_label, test_data, test_label):
        """        Save processed dataset
        """
        with open('data/bkg_data/train_data.pkl', 'wb') as handle:
            pickle.dump(BUtils.flatten_inputs(train_data), handle)
        with open('data/bkg_data/test_data.pkl', 'wb') as handle:
            pickle.dump(BUtils.flatten_inputs(test_data), handle)
        with open('data/bkg_data/train_label.pkl', 'wb') as handle:
            pickle.dump(BUtils.flatten_inputs(train_label), handle)
        with open('data/bkg_data/test_label.pkl', 'wb') as handle:
            pickle.dump(BUtils.flatten_inputs(test_label), handle)


class DataVis():
    """
    Class defines functions that perform data visualization before and after ML prediction
    """

    def __init__(self, features, labels, data_file_names, feature_col_names):
        self.features = features
        self.labels = labels
        # data_file_names from object of Pipeline class
        self.data_file_names = data_file_names
        # feature_col_names from object of Pipeline class
        self.feature_col_names = feature_col_names

    def xgboost_visualize_prediction(self, estimator, smooth_type: str, visualize_all=True, close_fig=False):
        """
        Visualize xgboost predictions with different smoothing methods

        """
        y_pred_lst = []
        fig_lst = []
        for i in range(len(self.features)):
            y_pred = estimator.predict(
                np.array(BUtils.flatten_inputs(self.features[i])))
            y_pred = BUtils.smooth_time_series_prediction(y_pred, smooth_type)

            y_pred_lst += [y_pred]

            if not visualize_all:
                fig = plt.figure()
                plt.plot(np.array(BUtils.flatten_inputs(
                    self.labels[i])), label='Measured')
                plt.plot(y_pred, label='Predicted')
                plt.xlabel('Num of Samples')
                plt.ylabel('Background (mV)')
                plt.title(
                    f'Experiment {os.path.basename(self.data_file_names[i])}', fontsize=12)
                plt.legend()
                fig_lst.append(fig)
                if close_fig:
                    plt.close()
                plt.show()

        if visualize_all:
            y_pred_lst = [_ for sublist in y_pred_lst for _ in sublist]
            fig = plt.figure()
            plt.plot(BUtils.flatten_inputs(
                self.labels, two_dim=True), label='Measured')
            plt.plot(y_pred_lst, label='Predicted')
            plt.xlabel('Num of Samples')
            plt.ylabel('Background (mV)')
            plt.title(f'Smooth type: {smooth_type}')
            plt.legend()
            if close_fig:
                plt.close()
            plt.show()
            return fig

        return fig_lst

    def xgboost_feature_importance(self, estimator):
        """
        Plot xgboost feature importance score

        Arguments:
        feature_col_names: feature_col_names from object of Pipeline()
        """
        fig = plt.figure(figsize=(8, 20))
        plt.barh(self.feature_col_names, estimator.feature_importances_, color='b', height=0.2)
        plt.xlabel('Importance score', fontsize=15)
        plt.ylabel('Input features', fontsize=15)
        plt.title('Feature importance', fontsize=24)
        plt.tight_layout()
        plt.show()
        return fig

    def visualize_features(self, close_fig=False):
        """
        Plot the histogram of the input features
        Arguments:
        data: Either train_features or test_features
        close_fig: whether to show figure in notebook or not

        Outputs:
        fig_lst: a list to store figure objects that can be stored in Mlflow artifacts    
        """
        # flatten train features into two dimensional array (n_sanmes, n_features)
        features_f = BUtils.flatten_inputs(self.features, two_dim=True)
        # use a list to store figure objects, which will be passed to MLFLOW
        fig_lst = []
        for i in range(len(self.feature_col_names)):
            fig = plt.figure()
            mu, sigma = scipy.stats.norm.fit(features_f[:, i])
            plt.hist(features_f[:, i], bins=100, alpha=0.5, color='blue')
            plt.title(f'mean: {round(mu, 3)}, and std: {round(sigma, 3)}')
            plt.axvline(x=mu)
            plt.xlabel(self.feature_col_names[i])
            plt.ylabel('Counts')
            fig_lst.append(fig)
            if close_fig:
                plt.close()
            plt.show()
        return fig_lst

    def visualize_labels(self, title):
        """
        Plot label histogram

        Arguments:
        labels: labels for illustration
        title: Titil for the plotted figure (e.g. 'Train labels distribution' or 'Test labels distribution')
        """
        input_labels = BUtils.flatten_inputs(self.labels).reshape(-1)
        fig = plt.figure()
        plt.hist(input_labels, bins=100, color='red')
        plt.title(title + f' from {len(self.labels)} experiments')
        plt.xlabel('Background (mV)')
        plt.ylabel('Counts')
        plt.show()
        return fig


class BUtils:
    """
    BUtils defines all utility functions for background ML
    """
    @staticmethod
    def flatten_inputs(inputs, two_dim=False):
        """
        Features and Labels are constructed with the ease to visualize per experiment, but training requires flatten it
        """
        flatten_input = np.array([i for sublist in inputs for i in sublist])
        if two_dim:
            flatten_input = flatten_input.reshape((len(flatten_input), -1))
        return flatten_input

    @staticmethod
    def smooth_time_series_prediction(y_pred, smooth_type: str):
        """
        Apply a median filter for the estimated background

        Arguments:
        y_pred: predicted values from model
        type: type of smooth to appy to time series prediction
        """
        if smooth_type == 'none':
            smooth_signal = y_pred
        if smooth_type == "med_avg":
            smooth_signal = signal.medfilt(y_pred, 3)
            avg_win = 15
            for i in range(avg_win, len(y_pred) - avg_win):
                smooth_signal[i] = np.mean(y_pred[i - avg_win:i + avg_win])
        if smooth_type == 'roc':
            smooth_signal = y_pred.copy()
            max_try = 0
            while len(np.where(abs(np.ediff1d(y_pred)) > 0.4)[0]) > 0 and max_try <= 100:
                smooth_idx = np.where(abs(np.ediff1d(y_pred)) > 0.4)[0]
                for i in smooth_idx:
                    smooth_signal[i + 1] = smooth_signal[i]
                max_try += 1
                y_pred = smooth_signal.copy()
        return smooth_signal

    @staticmethod
    def xgboost_report_best_scores(results, n_top=3):
        """
        Print out best scores from xgboost models
        """
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")        
class Trainer():
    """
    Trainer class defines training functions with an end-to-end pipeline
    """

    def __init__(self, experiment_paths, num_of_inputs, stride, exp_type) -> None:
        self.experiment_paths = experiment_paths
        self.exp_type = exp_type
        self.pipeline = Pipeline(self.experiment_paths, num_of_inputs, stride, self.exp_type)
        # define run_name
        self.FeatureEngineering_string = ''
        for key, val in DLConst.FEATURE_ENGINEERING_STEPS.items():
            if val:
                self.FeatureEngineering_string += key + ' * '
        # set MLFLOW experiment information
        self.run_name = self.exp_type + ' | ' + self.FeatureEngineering_string

        try:
            self.experiment_id = mlflow.create_experiment(DLConst.EXPERIMENT_NAME)
            print(f'Created new experiment: {DLConst.EXPERIMENT_NAME} with ID: {self.experiment_id}')
        except:
            print(f'{DLConst.EXPERIMENT_NAME} experiment already exists')
            self.experiment_id = mlflow.get_experiment_by_name(DLConst.EXPERIMENT_NAME).experiment_id

    def train(self, model_type, n_fold=5, n_iter=100, n_jobs=8):
        """
        Train xgboost based on cutomized cross validation on experiment base

        Arguments:
        model_type: the type of model to train:
                    xgb: xgboost regression
                    lr: linear regression
        """
        data_all, label_all = self.pipeline.input_pipeline()

        # instantiate a data_vis object from DataVis class
        data_vis = DataVis(data_all, label_all,
                           self.pipeline.data_file_names, self.pipeline.feature_col_names)
        train_feature_fig_lst = data_vis.visualize_features(close_fig=True)
        train_label_fig = data_vis.visualize_labels(
            title='All data labels distribution')

        # calculate number of files that are in the leave out set in the cross validation
        res = len(data_all) % n_fold  # e.g. 38 % 5 = 3
        parser_len = len(data_all) // n_fold  # e.g. 38 // 5 = 7

        # assign idx for each file/experiment
        groups = []
        for i in range(n_fold):
            if res > 0:
                temp_len = parser_len + 1
                res -= 1
            else:
                temp_len = parser_len
            for _ in range(temp_len):
                groups.append(i)
        groups = np.array(groups)

        # broadcast idx to all samples in each file/experiment
        group_idx = []
        file_sample_n_lst = [len(_) for _ in data_all]
        for i in range(len(file_sample_n_lst)):
            group_idx.append(file_sample_n_lst[i] * [groups[i]])

        group_idx = [_ for sublist in group_idx for _ in sublist]
        #print(f'Group index is : {group_idx}')
        #print(f'Length of group index is {len(group_idx)}')
        group_kfold = GroupKFold(n_splits=n_fold)

        print(f'Start training run : {self.run_name}')

        # flatten the last two dimensions for xgboost (e.g. (64, 1, 1) -> (64, 1))
        X = BUtils.flatten_inputs(data_all, two_dim=True)
        Y = BUtils.flatten_inputs(label_all, two_dim=True)
        print(f'Training data shape: {X.shape}')
        print(f'Training label shape: {Y.shape}')

        # instantiate xgboost
        if model_type == 'xgb':
            model = XGBRegressor()
            params = {
                "colsample_bytree": uniform(0.7, 0.3),
                "gamma": uniform(0, 0.5),
                "learning_rate": uniform(0.03, 0.3),  # default 0.1
                "max_depth": randint(2, 6),  # default 3
                "n_estimators": randint(100, 150),  # default 100
                "subsample": uniform(0.6, 0.4)
            }

            search = RandomizedSearchCV(model, param_distributions=params, random_state=47, n_iter=n_iter,
                                        scoring='neg_mean_absolute_error', cv=group_kfold, verbose=1, n_jobs=n_jobs,
                                        return_train_score=True, refit=True)
        elif model_type == 'lr':
            model = LinearRegression(normalize=True)
            params = {
                "fit_intercept": [True, False]
            }
            search = GridSearchCV(model, param_grid=params, scoring='neg_mean_absolute_error',
                                  cv=group_kfold, verbose=1, n_jobs=n_jobs,
                                  return_train_score=True, refit=True)

        else:
            print('Error: No model is selected for training!')
            return

        with mlflow.start_run(run_name=self.run_name + '| ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                              experiment_id=self.experiment_id):
            search.fit(X, Y, groups=group_idx)
            print('******************\n\n')
            # apply the trained model on test set
            trained_model = search.best_estimator_

            mlflow.log_params(search.best_params_)
            mlflow.log_metric('Avg CV MAE', search.best_score_)
            mlflow.sklearn.log_model(trained_model, "xgboost_model")

            # log train feature figures
            for i, train_feature_fig in enumerate(train_feature_fig_lst):
                mlflow.log_figure(train_feature_fig, 'feature_' + self.pipeline.feature_col_names[i] + '_fig.png')

            # log train label figure
            mlflow.log_figure(train_label_fig, 'train_label.png')

            # log feature engineering steps
            mlflow.log_text(self.FeatureEngineering_string, 'FEATURE_ENGINEERING_STEPS.txt')
            # Log features name that are used for training
            run_dict = {'Features': self.pipeline.feature_col_names}
            mlflow.log_dict(run_dict, 'input_features.yml')

            # Log training hyperparameters
            mlflow.log_dict(search.get_params(), 'training_param.yml')

            # log feature importance figure
            if model_type == "xgb":
                feature_imp = data_vis.xgboost_feature_importance(search.best_estimator_)
                mlflow.log_figure(feature_imp, 'feature_importance.png')

        mlflow.end_run()


class Evaluator():
    """
    Evaluator class defines evaluation functions with test dataset
    """

    def __init__(self, experiment_paths, model_path, num_of_inputs, stride, exp_type):
        self.experiment_paths = experiment_paths
        self.model_path = model_path
        self.exp_type = exp_type
        self.reset_FEATURE_ENGINEERING_STEPS()
        self.pipeline = Pipeline(
            self.experiment_paths, num_of_inputs, stride, self.exp_type)
        # defin run_name
        self.FeatureEngineering_string = ''
        for key, val in DLConst.FEATURE_ENGINEERING_STEPS.items():
            if val:
                self.FeatureEngineering_string += key + ' * '
        # set MLFLOW experiment information
        self.run_name = self.exp_type + ' | ' + self.FeatureEngineering_string

        try:
            self.experiment_id = mlflow.create_experiment(
                DLConst.EXPERIMENT_NAME + '-TEST')
            print(
                f"Created new experiment: {DLConst.EXPERIMENT_NAME} + '-TEST' with ID: {self.experiment_id}")
        except:
            print(f"{DLConst.EXPERIMENT_NAME} + '-TEST' experiment already exists")
            self.experiment_id = mlflow.get_experiment_by_name(
                DLConst.EXPERIMENT_NAME + '-TEST').experiment_id
        self.y_pred = None
        self.x_test = None
        self.y_test = None

    def evaluate_model(self):
        """
        Evaluate model with test data, and save the results in MLflow
        """
        # process test data files
        print('Transforming features...')
        test_features, test_labels = self.pipeline.input_pipeline()
        print(f'test feature shape: {test_features.shape}')
        print(f'test label shape: {test_labels.shape}')

        # instantiate a data_vis object from DataVis class
        data_vis = DataVis(test_features, test_labels,
                           self.pipeline.data_file_names, self.pipeline.feature_col_names)
        test_feature_fig_lst = data_vis.visualize_features(close_fig=True)
        test_label_fig = data_vis.visualize_labels(
            title='All data labels distribution')

        self.x_test = BUtils.flatten_inputs(test_features, two_dim=True)
        self.y_test = BUtils.flatten_inputs(test_labels, two_dim=True)
        print(f'Test data shape: {self.x_test.shape}')
        print(f'Test label shape: {self.y_test.shape}')

        # load model
        trained_model = pickle.load(open(self.model_path, 'rb'))
        print('Make prediction...')
        self.y_pred = trained_model.predict(self.x_test)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        print(f'Mean absolute error on test data is: {round(mae, 3)}')

        smooth_med_avg_y_pred = BUtils.smooth_time_series_prediction(
            self.y_pred, smooth_type='med_avg')
        smooth_med_avg_mae = mean_absolute_error(
            self.y_test, smooth_med_avg_y_pred)
        print(
            f'Median-Avg smoothed Mean absolute error on test data is: {round(smooth_med_avg_mae, 3)}')

        with mlflow.start_run(run_name=self.run_name + '| ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                              experiment_id=self.experiment_id):
            print('log MAE')
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('Smooth_median_mae', smooth_med_avg_mae)
            mlflow.sklearn.log_model(trained_model, "xgboost_model")

            # log feature engineering steps
            mlflow.log_text(self.FeatureEngineering_string,
                            'test_FEATURE_ENGINEERING_STEPS.txt')
            # Log features name that are used for training
            run_dict = {'Features': self.pipeline.feature_col_names}
            mlflow.log_dict(run_dict, 'test_input_features.yml')

            # log test feature figures
            for i, test_feature_fig in enumerate(test_feature_fig_lst):
                mlflow.log_figure(
                    test_feature_fig, 'test_feature_' + self.pipeline.feature_col_names[i] + '_fig.png')

            # log test label figure
            mlflow.log_figure(test_label_fig, 'test_label.png')

            # Log plot from all prediction
            # without smoothing
            visual_all_exp = data_vis.xgboost_visualize_prediction(
                trained_model, smooth_type='none', visualize_all=True)
            mlflow.log_figure(visual_all_exp, 'visual_results_all.png')

            # with median filter
            visual_all_exp_smooth_med_avg = data_vis.xgboost_visualize_prediction(
                trained_model, smooth_type='med_avg', visualize_all=True)
            mlflow.log_figure(visual_all_exp_smooth_med_avg,
                              'visual_results_all_smooth_median_avg.png')

            if len(test_features) > 1:
                print('log individual test results')
                # Log plot from individual prediction
                # without smoothing
                visual_per_exp_lst = data_vis.xgboost_visualize_prediction(
                    trained_model, smooth_type='none', visualize_all=False)
                for i, visual_per_exp in enumerate(visual_per_exp_lst):
                    mlflow.log_figure(
                        visual_per_exp, 'visual_per_experiment_' + str(i) + '.png')

                # with median filter
                visual_per_exp_median_avg_lst = data_vis.xgboost_visualize_prediction(
                    trained_model, smooth_type='med_avg', visualize_all=False)
                for i, visual_per_exp_median_avg in enumerate(visual_per_exp_median_avg_lst):
                    mlflow.log_figure(
                        visual_per_exp_median_avg, 'visual_per_experiment_smooth_med_avg_' + str(i) + '.png')

        mlflow.end_run()

    def reset_FEATURE_ENGINEERING_STEPS(self):
        # set all feature engineering steps to false
        for key, _ in DLConst.FEATURE_ENGINEERING_STEPS.items():
            DLConst.FEATURE_ENGINEERING_STEPS[key] = False
        # reset feature engineering steps based on the trained model
        feature_eng_steps_df = pd.read_csv(os.path.join(Path(self.model_path).parent.parent,
                                                        'FEATURE_ENGINEERING_STEPS.txt'),
                                           delimiter="\t", header=None)
        feature_eng_steps = feature_eng_steps_df.iloc[0, 0].split(' * ')
        for feature_eng_step in feature_eng_steps:
            if feature_eng_step in DLConst.FEATURE_ENGINEERING_STEPS:
                DLConst.FEATURE_ENGINEERING_STEPS[feature_eng_step] = True
        print(DLConst.FEATURE_ENGINEERING_STEPS)


# if __name__ == '__main__':
#     # TODO: convert those to allow user input from terminal
#     # training hyperparameters
#     NUM_OF_INPUTS = 1
#     STRIDE = 1
#     TRAIN_RATIO = 0.9
#     EXP_TYPE = 'p'

#     for acc_trans_bool in [True, False]:
#         DLConst.FEATURE_ENGINEERING_STEPS['acc_trans'] = acc_trans_bool
#         for led_pd_trans_bool in [True, False]:
#             DLConst.FEATURE_ENGINEERING_STEPS['led_pd_trans'] = led_pd_trans_bool
#             for led_pd_all_bool in [True, False]:
#                 DLConst.FEATURE_ENGINEERING_STEPS['led_pd_all'] = led_pd_all_bool
#                 for led_pd_var_bool in [True, False]:
#                     DLConst.FEATURE_ENGINEERING_STEPS['led_pd_var'] = led_pd_var_bool
#                     for opt_property_bool in [True, False]:
#                         DLConst.FEATURE_ENGINEERING_STEPS['opt_property'] = opt_property_bool
#                         try:
#                             pig_bkg_file_lst = DataImporter.find_background_files_pig()
#                             adhesive_exp_paths, long_exp_paths, short_exp_paths = DataImporter.find_background_from_local()
#                             bkg_file_lst = DataImporter.find_background_from_clinical(
#                                 site_filter="", day_post_injection_filter=0, bkg_std_filter=3)
#                             all_exp_list = [pig_bkg_file_lst]
#                             trainer = Trainer(
#                                 all_exp_list, NUM_OF_INPUTS, STRIDE, TRAIN_RATIO, EXP_TYPE)
#                             trainer.train_xgboost()
#                         except:
#                             print(f"Can't perform model training")
#                             raise
