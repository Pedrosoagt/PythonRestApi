import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from pmdarima.arima import auto_arima
import pmdarima as pm
import matplotlib.pyplot as plt

class PredictionSystem():
    """
    Prediction System implementation
    
    Example of usage:
    
    $ pred_sys = PredictionSystem(boards_list[0:300], boards_list[300:400], feat='Volume')
    $ pred_sys.identify_risk_regions(board_section=False, n_comp=20)
    $ pred_sys.apply_PCA(scale=False, optimize=True, method='percentile', percentile=[5,75], val_range=[0.1,1], step=0.01, beta=0.3)
    $ pred_sys.apply_ARIMA()
    $ pred_sys.evaluate()
        
    
    """
    def __init__(self, boards_list, test_boards, feat='Volume'):
        """ Init method.
        
        Stores the list of panels for 'train' and 'test', and the feature 
        
        Args:
            - boards_list: list of panels to be used on the trainig process. Must be previously ordered by datetime.
            - test_boards: list of panels to be used on the test/validation process. Must be previously ordered by datetime.
            - feat: contains the feature to be analyzed: 'Volume', 'ShortPct', 'LongPct' (not sure if it works for 'Area', 'Height')
        
        """
        # Feature to be analyzed along the process
        self.feat = feat
        
        # Store previously ordered array containing boards list for train / test
        self.boards_list = boards_list
        self.test_boards = test_boards
        
        # Flag to denote if analyzed feature is ShortPct or LongPct
        if feat in ['ShortPct', 'LongPct']:
            self.isPct = True
        else:
            self.isPct = False
        
        # Labels/Status for train list of boards, test list of boards
        self.train_boards_status = self.__get_status_boards(self.boards_list)
        self.test_boards_status = self.__get_status_boards(self.test_boards)
    
    ### Part 1: Identify risk regions ###
    
    def __get_pad_region(self, n=3):
        """ Divides the boards into sections, looking at (X, Y) coordinates from every pad
        """
        board = self.boards_list[0]
        Xs = []
        Ys = []
        for p in board['Pad'].keys():
            Xs.append(float(board['Pad'][p]['X']))
            Ys.append(float(board['Pad'][p]['Y']))

        Xmax, Xmin, Ymax, Ymin = np.max(Xs), np.min(Xs), np.max(Ys), np.min(Ys)

        xdiv = np.linspace(Xmin, Xmax, num=n)
        ydiv = np.linspace(Ymin, Ymax, num=n)

        divisions = []
        for i in range(1, len(xdiv)):
            for j in range(1, len(ydiv)):
                divisions.append([xdiv[i-1], xdiv[i], ydiv[j-1], ydiv[j]])
    
        pads_total = []
        for group in divisions:
            pads_region = []
            for p in board['Pad'].keys():
                x = float(board['Pad'][p]['X'])
                y = float(board['Pad'][p]['Y'])

                if (x < group[1] and x > group[0]) and (y < group[3] and y > group[2]):
                    pads_region.append(p)

            pads_total.append(pads_region)

        return pads_total
    
    def __most_fail_pads(self):
        """ Calculates a dictionary of fail occurrences for each pad. Might need improvement in performance/data structure
        """
        fail_pads = {}

        for k in self.boards_list[0]['Pad'].keys():
            if self.isPct:
                flag = False
            else:
                flag = True

            fail_dict = {}
            dates = []
            failed = []
            lack_xs = []

            for b in self.boards_list:
                if b['Pad'][k][self.feat]['Status'] == 'F':
                    dates.append(b['Datetime'])
                    if flag:
                        if float(b['Pad'][k][self.feat]['Value']) > float(b['Pad'][k][self.feat]['UpFail']):
                            lack_xs.append('Up')
                        else:
                            lack_xs.append('Down')

            if len(dates) != 0:
                fail_dict['Count'] = len(dates)
                if flag:
                    ctr = Counter(lack_xs)
                    if 'Up' in ctr.keys():
                        fail_dict['Excess'] = ctr['Up']
                    else:
                        fail_dict['Excess'] = 0
                    if 'Down' in ctr.keys():
                        fail_dict['Lack'] = ctr['Down']
                    else:
                        fail_dict['Lack'] = 0
                fail_dict['Dates'] = dates
                fail_pads[k] = fail_dict

        return fail_pads
    
    def __get_most_fail_components(self, n=10):
        """ Calculates the most failing components. Might need improvement.
        """
        board = self.boards_list[0]
        self.mfp = self.__most_fail_pads()

        count = []
        pad = []

        for k in self.mfp.keys():
            count.append(self.mfp[k]['Count'])
            pad.append(k)

        comp = [board['Pad'][c]['ComponentId'] for c in pad]
        df = pd.DataFrame(data={'Component': comp, 'Pad': pad, 'Count': count})
        comps = np.array(df.sort_values(by=['Count'], ascending=False)['Component'].unique())

        if n > len(comps):
            n = len(comps)
        
        return comps[:n]
    
    def __get_risk_regions(self, n=10):
        """ Uses the 'risk region' method based on fail occurrences over training panels. Might need improvement.
        """
        print("Identify risk regions!")
        most_fail_comp = self.__get_most_fail_components(n)

        print("Most Fail Components:")
        pprint(most_fail_comp)

        pdd = [p for p in self.boards_list[0]['Pad'].keys()\
               if self.boards_list[0]['Pad'][p]['ComponentId'] in most_fail_comp]
        print("Affected Pads:")
        pprint(pdd)
        
        return pdd
    
    def identify_risk_regions(self, board_section=False, n_comp=10):
        """ Main method to identify risk regions of the board
        
            Args:
                - board_section: Boolean, indicates the methodology to identify regions:
                    - True: Section the boards spatially into 4 parts
                    - False: Use the 'risk region' method, based on pads/components that fail the most
                        - n_comp: number of top fail components used to define the 'risk region' - Might need improvements later.
                    Store the list of pads obtained in the process into instance variables
        
        """
        if board_section:
            pads_list_total = self.__get_pad_region(3)
        else:
            pads_list_total = []
            pads_list_total.append(self.__get_risk_regions(n=n_comp))
        
        self.pads_list_total = pads_list_total
    
    
    ### Part 2: Apply PCA ###
    
    def __get_status_boards(self, bd_list):
        """ Gets the status of each panel: if it contains at least one failure on any pad, it is classified as 'F', otherwise, 'P'
        
            Args:
                - bd_list: list containing the panels
            Returns:
                - status_boards: list containing the status result for every panel from bd_list. Ex: ['P', 'P', 'F', ...]
        """
        status_boards = []
        
        for b in bd_list:
            count_fails = b['Pad_info']['Fail'][self.feat]

            if count_fails > 0:
                status_boards.append('F')
            else:
                status_boards.append('P')
        
        return status_boards
    
    def __get_boards_values(self, bd_list, pads_list):
        """ Gets main infos about the list of panels (bd_list) regarding the list of pads (pads_list)
        
            Args:
                - bd_list: list containing the panels
                - pads_list: list containing the pads used for analysis
            Returns:
                 Sklearn compatible structures
                - values_total: list containing the values of selected feature for each pad (pads_list) of each board (bd_list).
                - max_lim_total: list containing the max limit of selected feature for each pad (pads_list) of each board (bd_list)
                - min_lim_total: list containig the min limit of selected feature for each pad (pads_list) of each board (bd_list)
                - status_chosen_pads: list containing the status ('F', 'P') for each pad (pads_list) of each board (bd_list)
        """
    
        # Iterate through the boards on boards list (train set)
        values_total = []
        max_lim_total = []
        min_lim_total = []
        status_chosen_pads = []
        
        for b in bd_list:
            values = []
            max_lim = []
            min_lim = []
            status_pads = []

            for p in pads_list:
                values.append(float(b['Pad'][str(p)][self.feat]['Value']))
                if self.isPct:
                    max_lim.append(float(b['Pad'][str(p)][self.feat]['Limit']))
                    min_lim.append(0)
                else:
                    max_lim.append(float(b['Pad'][str(p)][self.feat]['UpFail']))
                    min_lim.append(float(b['Pad'][str(p)][self.feat]['LowFail']))
                status_pads.append(b['Pad'][str(p)][self.feat]['Status'])

            values_total.append(values)
            max_lim_total.append(max_lim)
            min_lim_total.append(min_lim)
            status_chosen_pads.append(status_pads)
    
        return values_total, max_lim_total, min_lim_total, status_chosen_pads

    def __apply_pca_to_values(self, values, maxlimit, minlimit, scale=False):
        """ Applies PCA transformation
        
            Args:
                - values: trainig values, might be the values obtained
                - maxlimit: list containing the values of maximum threshold
                - minlimit: list containing the values of minimum threshold
                - scale: Boolean. Indicates if StandardScaler will be used
        """
        print("Apply PCA!")
        if scale:
            sc = StandardScaler()
            values = sc.fit_transform(values)
            maxlimit = sc.transform(maxlimit)
            minlimit = sc.transform(minlimit)

        pca_vol = PCA(n_components=1)
        pca_vol.fit(values)
        
        vol_t = pca_vol.transform(values)
        max_t = pca_vol.transform(np.array(maxlimit))
        min_t = pca_vol.transform(np.array(minlimit))

        print("Explained variance: ", pca_vol.explained_variance_ratio_)

        return vol_t, max_t, min_t, pca_vol
    
    def __optimize_limits(self, val_t, max_t, min_t, actual_status, method='search', percentile=[25,75],\
                          val_range=[0.1,1], step=0.1, beta=0.5):
        """ Limit optimization methods.
        
            Args:
                - val_t: list of PCA transformed values (train) to be tested on optimization
                - max_t: list of PCA transformed higher threshold
                - min_t: list of PCA transformed lower threshold
                - actual_status: list containing the actual status of training panels. Used in 'search' optimization method
                - method:
                    - 'search' : grid search over a factor, straightening both lower and higher thresolds at same time
                        - val_range: list [min_multiplier, max_multiplier]
                        - step: values that indicates each step taken in evaluating the threshold, between 'min_multiplier' and 'max_multiplier'
                        - beta: the beta parameter of 'f_beta score', used as metric of evaluation in the parameter search.
                    - 'percentile': calculates the percentile for lower and higher limits, use them as threshold.
                        - percentile: list [low_percetile, high_percentile] indicates the percentile used for definig the thresholds.
            Returns:
                - max_lim: optimized higher threshold
                - min_lim: optimized lower threshold
                - best_pca_eval: list of status of traning boards obtained from the optimization
        """
        if method == 'search':
            best_f2 = 0
            best_op = 0

            for op in np.flip(np.arange(val_range[0], val_range[1], step)):
                if self.isPct:
                    evaluated_status = ['F' if (val_t[i][0] > op*max_t[i][0] or val_t[i][0] < min_t[i][0])\
                                        else 'P' for i in range(len(val_t))]
                else:
                    evaluated_status = ['F' if val_t[i][0] > op*max_t[i][0] or val_t[i][0] < op*min_t[i][0]\
                                        else 'P' for i in range(len(val_t))]

                f2_new = fbeta_score([1 if el is 'P' else 0 for el in actual_status],\
                                     [1 if el is 'P' else 0 for el in evaluated_status], beta=beta)
                if best_f2 < f2_new:
                    best_f2 = f2_new
                    best_op = op
                    best_pca_eval = evaluated_status
                    if self.isPct:
                        max_lim = best_op*max_t
                        min_lim = min_t
                    else:
                        max_lim = best_op*max_t
                        min_lim = best_op*min_t
             
        elif method == 'percentile':
            p_max = np.percentile(val_t, percentile[1])
            p_min = np.percentile(val_t, percentile[0])
            max_lim = len(max_t)*[[p_max]]
            min_lim = len(min_t)*[[p_min]]
            best_pca_eval = evaluated_status = ['F' if val_t[i][0] > max_lim[0][0] or val_t[i][0] < min_lim[0][0] else 'P'\
                                                for i in range(len(val_t))]
        
        return max_lim, min_lim, best_pca_eval
    
    def apply_PCA(self, scale=False, optimize=True, method='search', percentile=[25,75], val_range=[0.1,1], step=0.01, beta=0.5):
        """ Main method used on apllying PCA to train panels list, optimizing the thresholds.
            Also stores the PCA transformation (obtained by applying PCA to train set), then apply it to test values list. 
            Stores the results on instance variables.
        
            Args:
            - method:
                - 'search' : grid search over a factor, straightening both lower and higher thresolds at same time
                    - val_range: list [min_multiplier, max_multiplier]
                    - step: values that indicates each step taken in evaluating the threshold, between 'min_multiplier' and 'max_multiplier'
                    - beta: the beta parameter of 'f_beta score', used as metric of evaluation in the parameter search.
            - 'percentile': calculates the percentile for lower and higher limits, use them as threshold.
                    - percentile: list [low_percetile, high_percentile] indicates the percentile used for definig the thresholds.
        
        """
        
        # This lists contain the general information of the board (independent of the strategy of board segmentation)
        self.train_values_total = []
        self.test_values_total = []
        self.max_limit_total = []
        self.min_limit_total = []
        self.pca_transform = []
        
        self.train_status_total = []

        # Iterate through the pads_list_total (which represents the segments of the boards)
        for pads_list in self.pads_list_total:
            # Get lists of values from the boards list 
            values_list, max_lim_list, min_lim_list, status_pads_list = self.__get_boards_values(self.boards_list, pads_list)

            df = pd.DataFrame(data=values_list, columns=pads_list)

            # Apply PCA and get results
            val_t, max_t, min_t, pca_val = self.__apply_pca_to_values(values_list, max_lim_list, min_lim_list, scale=scale)
            print("")
            
            # Save main results into instance
            self.train_values_total.append(val_t)
            self.pca_transform.append(pca_val)

            if max_t[0] < min_t[0]:
                aux = max_t
                max_t = min_t
                min_t = aux
            
            status_chosen_pads = ['F' if 'F' in pd_lst else 'P' for pd_lst in status_pads_list]
            
            if optimize:
                max_t_new, min_t_new, best_pca_eval = self.__optimize_limits(val_t, max_t, min_t, status_chosen_pads,\
                                                                             val_range=val_range, step=step, beta=beta,\
                                                                             percentile=percentile, method=method)
            else:
                max_t_new, min_t_new = max_t, min_t
                best_pca_eval = ['F' if (val_t[i][0] > max_t[i][0] or val_t[i][0] < min_t[i][0])\
                                 else 'P' for n in range(len(val_t))]
                
            self.max_limit_total.append(max_t_new)
            self.min_limit_total.append(min_t_new)
            self.train_status_total.append(best_pca_eval)

            # Get lists of values from the boards list 
            test_values_list, _1, _2, _3 = self.__get_boards_values(self.test_boards, pads_list)

            test_values_t = pca_val.transform(test_values_list)
            self.test_values_total.append(test_values_t)

    ### Part 3: Apply ARIMA and Prediction
    
    def __get_arima(self, val_t, next_vals, n_periods=10):
        """ Obtains the best ARIMA fit for the train set, and apply it to obtain predictions.
            Also contain alternative strategies for obtaining best ARIMA. Can be improved.
            
            Args:
                - val_t: list of values of PCA transformed train set.
                - next_vals: list of PCA transformed test set, from real data. This are the values that ARIMA will try to predict.
                - n_periods: number of predictions that ARIMA model will make.
            Returns:
                - prediction: list of predicted values (resulting from ARIMA predictions)
                - conf_interval: list of confidence interval for each predicted value
                - AA: the best ARIMA instance
        """
        
        AA = auto_arima(val_t, information_criterion='aic', seasonal=False, suppress_warnings=True)
        print("")
        print("************************")
        print("ARIMA results:")
        #AA.plot_diagnostics(figsize=(10, 7))
        print(AA.summary())
        print("************************")
        print("")

        prediction = []
        conf_interval = []

    ######### Run ARIMA on each new sample
#         pr, conf = AA.predict(n_periods=1, return_conf_int=True)
#         prediction.append(pr)
#         conf_interval.append(conf)

#         vol_t_new = vol_t

#         for n in next_vals:
#             vol_t_new = np.append(vol_t_new, [n])
#             AA = auto_arima(vol_t_new, information_criterion='aic', seasonal=False)
#             pr, conf = AA.predict(n_periods=1, return_conf_int=True)
#             prediction.append(pr)
#             conf_interval.append(conf)

    ########## Add only new observations:
        pr, conf = AA.predict(n_periods=1, return_conf_int=True)
        prediction.append(pr)
        conf_interval.append(conf)

        for n in next_vals:
            AA.add_new_observations([n])
            pr, conf = AA.predict(n_periods=1, return_conf_int=True)
            prediction.append(pr)
            conf_interval.append(conf)

    ########## Just fit ARIMA once
#         prediction, conf_interval = AA.predict(n_periods=n_periods, return_conf_int=True)

        return prediction, conf_interval, AA
    
    def plot_prediction(self, val, max_lim, min_lim, next_vals, pred, confid, feat):
        """ Function to plot the PCA transformed values from train, test, and prediction (along with confidence interval).
            Uses Plotly library.
            
            Args:
                - val: list of train values
                - max_lim: list of higher thresholds
                - min_lim: list of lower thresholds
                - next_vals: list of test values
                - pred: list of predictions
                - confid: array of groups of confidence intervals for predictions
                - feat: indicates which is the feature used for generating this plot
        """
        if len(next_vals) > len(pred):
            max_lenght = len(next_vals)
        else: 
            max_lenght = len(pred)

        maxl = max_lenght*[max_lim[0]]
        minl = max_lenght*[min_lim[0]]

        max_lim = list(max_lim) + maxl
        min_lim = list(min_lim) + minl

        X1 = [x for x in range(len(max_lim))]

        data = []

        main_feat = go.Scatter(
            x = X1[:len(val)],
            y = [v[0] for v in val],
            mode = 'lines+markers',
            name = feat,
        )
        data.append(main_feat)

        prediction = go.Scatter(
            x = X1[len(val):len(val)+len(pred)],
            y = [p[0] for p in pred],
            mode = 'lines+markers',
            name = 'Prediction',
            line=dict(color='rgb(0,176,246)')
        )
        data.append(prediction)

        confid1 = [c[0][0] for c in confid]
        confid2 = [c[0][1] for c in confid[::-1]]
        confid_ = confid1 + confid2

        confid_.insert(0, 0)
        confid_.append(0)

        X_pred1 = X1[len(val):len(val)+len(pred)]
        X_pred2 = X_pred1[::-1]
        X_pred_ = X_pred1 + X_pred2

        X_pred_.insert(0, len(val)-0.0001)
        X_pred_.append(len(val)-0.0001)

        trace1 = go.Scatter(
            x= X_pred_,
            y= confid_,
            fill='tozerox',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            #name='Fair',
        )
        data.append(trace1)

        nxt = go.Scatter(
            x = X1[len(val):len(val)+len(next_vals)],
            y = [n[0] for n in next_vals],
            mode = 'lines+markers',
            name = "Next Values",
        )
        data.append(nxt)

        up_l = go.Scatter(
            x = X1,
            y = [m[0] for m in max_lim],
            mode = 'lines',
            name = 'Higher limit'
        )
        data.append(up_l)

        down_l = go.Scatter(
            x = X1,
            y = [m[0] for m in min_lim],
            mode = 'lines',
            name = 'Lower limit'
        )
        data.append(down_l)

        layout = go.Layout(
            title = 'Prediction for feature: {0}'.format(feat)
        )

        fig = go.Figure(data = data, layout = layout)
        iplot(fig)
    
    def apply_ARIMA(self):
        """ Main method to apply ARIMA to the obtained series. The values used are on instance variables previously obtained.
        """
        n_periods = len(self.test_boards)
        
        self.prediction_values_total = []
        self.prediction_conf_interval = []
        self.arima_model = []
        
        self.prediction_status_total = []
        
        for n in range(len(self.pads_list_total)):
            print(n)
            prediction_result, conf_interval, arima_model = self.__get_arima(self.train_values_total[n],\
                                                                             self.test_values_total[n],\
                                                                             n_periods)
            self.prediction_values_total.append(prediction_result)
            self.prediction_conf_interval.append(conf_interval)
            self.arima_model.append(arima_model)
            
            maxlim = self.max_limit_total[n]
            minlim = self.min_limit_total[n]
            
            status_prediction = ['F' if (prediction_result[i][0] > maxlim[0] or\
                                         prediction_result[i][0] < minlim[0]) else 'P'\
                                 for i in range(len(prediction_result))]
            
            self.prediction_status_total.append(status_prediction)
            
            self.plot_prediction(self.train_values_total[n], maxlim, minlim,\
                            self.test_values_total[n], prediction_result, conf_interval, self.feat)
        
    ### Part 4: Evaluation ###
    
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """ This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
    
    def evaluate(self):
        """ This methods runs the evaluation process of the overall prediction system.
            It displays the classification report from atcual values (real status of the panels) vs. prediction results.
            It also gives the confusion matrix of the results.
            It consumes values from instance variables previously calculated.
        """
        fail_idx = set([])
        for n in range(len(self.prediction_status_total)):
            fail_idx.update([i for i, x in enumerate(self.prediction_status_total[n]) if x == 'F'])
        
        self.prediction_status_final = ['F' if x in fail_idx else 'P' for x in range(len(self.prediction_status_total[0]))]
        
        print(classification_report(self.test_boards_status, self.prediction_status_final[:len(self.prediction_status_final)-1]))
        self.plot_confusion_matrix(confusion_matrix(self.test_boards_status, self.prediction_status_final[:len(self.prediction_status_final)-1],\
                                                    labels=['P', 'F']), ['P', 'F'], title="Actual results vs. Prediction Results",)
