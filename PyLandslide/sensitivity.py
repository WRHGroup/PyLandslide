#loading neccessery packages
import os
import json
import numpy as np
import pandas as pd
from osgeo import gdal
import logging
import random
import copy
import time
import string  
from datetime import datetime
logger = logging.getLogger(__name__)


class SensitivityEstimator(object):
    def __init__(self, json_file, trials, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_file = json_file
        self.trials = trials

    def load_data_from_json(self, **kwargs):
        """Load data from a file
        """
        data = os.path.join(os.getcwd(), self.json_file)
        if isinstance(data, str):
            logging.info('Loading data from file: "{}"'.format(data))
            with open(data, "r") as f:
                loaded_file = json.load(f)

        if loaded_file.get('factors') is None:
            raise ValueError('factors has not been found in the JSON file')
        self.factors = loaded_file.pop('factors')

        if loaded_file.get('susceptibility_classes') is None:
            raise ValueError('susceptibility_classes has not been found in the JSON file')
        self.susceptibility_classes = loaded_file.pop('susceptibility_classes')

        if loaded_file.get('output_directory') is None:
            print('output directory set to the cwd. If you wish to change this default value, add output_directory to the inputs provided in the JSON file')
            self.output_directory = os.getcwd()
        else:
            self.output_directory = os.path.join(os.getcwd(), loaded_file.pop('output_directory'))

    def factor_data_preperation(self, factors):
        keys = []
        names = []
        files = []
        weight_upper_bounds = []
        weight_lower_bounds = []
        for f in factors:
            keys.append(("weight_"+f["name"]))
            names.append(f["name"])
            files.append(f["file"])
            weight_upper_bounds.append(f["weight_upper_bound"])
            weight_lower_bounds.append(f["weight_lower_bound"])
        return keys, names, files, weight_lower_bounds, weight_upper_bounds

    def susceptibility_classes_data_preperation(self, susceptibility_classes):
        names = []
        class_upper_bounds = []
        class_lower_bounds = []
        for s in susceptibility_classes:
            names.append(s["name"])
            class_upper_bounds.append(s["class_upper_bound"])
            class_lower_bounds.append(s["class_lower_bound"])
        return names, class_lower_bounds, class_upper_bounds

    def check_weight_upper_and_lower_bounds(self, upper, lower):
        if max(upper)>1 or min(upper)<0 or max(lower)>1 or min(lower)<0:
            raise ValueError('weight_upper_bound and weight_lower_bound must be <= 1 and >= 0')
        for v,vv in enumerate(upper):
            if lower[v]>= upper[v]:
                raise ValueError('weight_upper_bound for each factor must be greater than weight_lower_bound')

    def check_class_upper_and_lower_bounds(self, upper, lower):
        for v,vv in enumerate(upper):
            if lower[v]>= upper[v]:
                raise ValueError('class_upper_bound for each factor must be greater than class_lower_bound')


    def create_results_dict(self):
        results_dic = {}
        for n in self.susceptibility_classes_names:
            results_dic[n]=[]
        for n in self.factor_weight_keys:
            results_dic[n]=[]
        return results_dic

    def generate_random_weights(self):
        temp_list1 = []
        random_weights = []
        for i, ii in enumerate(self.factor_weight_keys):
            temp_list1.append(random.uniform(self.factor_weight_lower_bounds[i], self.factor_weight_upper_bounds[i]))
        for m, k in enumerate(self.factor_weight_keys):
            random_weights.append(round(temp_list1[m]/sum(temp_list1),3))
        return random_weights

    def check_weight_ranges(self, weights):
        flag_list = []
        for w, ww in enumerate(weights):
            if ww >= self.factor_weight_lower_bounds[w] and ww <= self.factor_weight_upper_bounds[w]:
                flag_list.append(0)
            else:
                flag_list.append(1)
        
        if max(flag_list)==0:
            print("Weights generated")
            return True
        else:
            return False

    def overlay_factors(self, factor_weights):
        print("Overlaying factors...")
        alphabets = list(string.ascii_uppercase)
        calc_formula=""
        for w, weight_value in enumerate(factor_weights):
            if w == 0:
                calc_add = "("+alphabets[w]+"*"+str(weight_value)+")"
                calc_formula+=calc_add
            else:
                calc_add = "+("+alphabets[w]+"*"+str(weight_value)+")"
                calc_formula+=calc_add 

        full_calculation_command = "gdal_calc.py --quiet --overwrite --extent=union --outfile "+self.output_directory+"/lss.tif"
        for f, factor_file_dir in enumerate(self.factor_files):
            cmd_add = " -"+ alphabets[f] + " "+ factor_file_dir
            full_calculation_command+=cmd_add 

        full_calculation_command+= (" --calc="+calc_formula)
        os.system(full_calculation_command)

        lss_dataset = self.load_dataset(file_path=os.path.join(self.output_directory,"lss.tif"))
        lss_np = self.raster_to_numpy(raster_dataset=lss_dataset, no_data_value=lss_dataset.GetRasterBand(1).GetNoDataValue())

        susceptibility_class_pixels = self.number_of_pixels_in_susceptibility_classes(susceptibility_raster=lss_np)

        return susceptibility_class_pixels

    def load_dataset(self, file_path):
        print("Loading a locally saved raster file")
        path = os.path.join(os.getcwd(), file_path)
        raster_dataset = gdal.Open(path)
        return raster_dataset

    def raster_to_numpy(self, raster_dataset, no_data_value):
        print("Converting raster file to numpy array")
        raster_numpy = np.array(raster_dataset.GetRasterBand(1).ReadAsArray()).astype(float)
        raster_numpy[raster_numpy==no_data_value]=np.nan
        return raster_numpy

    def number_of_pixels_in_susceptibility_classes(self, susceptibility_raster):
        print("Classifying pixels based on susceptibility ranges")
        class_pixels = []
        for i in self.susceptibility_classes_names:
            class_pixels.append(0)

        total_number_of_pixels = (susceptibility_raster >= 0).sum()

        for c, cn in enumerate(self.susceptibility_classes_names):
            class_pixels[c] = ((self.susceptibility_classes_lower_bounds[c] <= susceptibility_raster) & (susceptibility_raster <= self.susceptibility_classes_upper_bounds[c])).sum()/total_number_of_pixels

        return class_pixels

    def execute(self):
        factor_data = self.factor_data_preperation(factors=self.factors)
        self.factor_weight_keys = factor_data[0]
        self.factor_names = factor_data[1]
        self.factor_files = factor_data[2]
        self.factor_weight_lower_bounds = factor_data[3]
        self.factor_weight_upper_bounds = factor_data[4]

        self.check_weight_upper_and_lower_bounds(upper=self.factor_weight_upper_bounds,lower=self.factor_weight_lower_bounds)

        susceptibility_classes_data = self.susceptibility_classes_data_preperation(susceptibility_classes=self.susceptibility_classes)
        self.susceptibility_classes_names = susceptibility_classes_data[0]
        self.susceptibility_classes_lower_bounds = susceptibility_classes_data[1]
        self.susceptibility_classes_upper_bounds = susceptibility_classes_data[2]

        self.check_class_upper_and_lower_bounds(upper=self.susceptibility_classes_upper_bounds,lower=self.susceptibility_classes_lower_bounds)

        #create results dic
        final_results_dic = self.create_results_dict()

        successful_trials = 0
        while successful_trials < self.trials:
            t0 = time.time()
            weight_inputs = self.generate_random_weights()
            if self.check_weight_ranges(weights=weight_inputs):
                print("============================================ starting trial "+str(successful_trials+1))
                susceptability_assessment = self.overlay_factors(factor_weights=weight_inputs)
                for sc, susceptibility_class in enumerate(self.susceptibility_classes_names):
                    final_results_dic[susceptibility_class].append(susceptability_assessment[sc])
                for wc, factor_weight_key in enumerate(self.factor_weight_keys):
                    final_results_dic[factor_weight_key].append(weight_inputs[wc])

                percent = (successful_trials+1)/self.trials*100
                print("time taken in the trial", round((time.time()-t0),0), "seconds|progress: (trial =",successful_trials+1,") (percentage =", round(percent,2), "%)")
                successful_trials += 1  

            #write the results
            final_result_pd = pd.DataFrame.from_dict(final_results_dic)
            final_result_pd.index.name = 'id'
            fn = os.path.join(self.output_directory, 'sensitivity_results.csv')
            final_result_pd.to_csv(fn)

        os.remove('temp_lss.tif')
        print("process completed.")

    def setup(self):
        self.load_data_from_json()
        logger.info('Setting up WeightRangeEstimator based on the file: "{}"'.format(self.json_file))
