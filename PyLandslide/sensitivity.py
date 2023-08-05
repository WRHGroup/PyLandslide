#loading neccessery packages
import os
import json
import numpy as np
import pandas as pd
import logging
import random
import copy
import time
import string  
import rasterio
from datetime import datetime
logger = logging.getLogger(__name__)


class SensitivityEstimator(object):
    def __init__(self, json_file, trials=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_file = json_file
        self.trials = trials

    def load_data_from_json(self, **kwargs):
        """Load data from a file
        """
        data = os.path.normpath(os.path.join(os.getcwd(), self.json_file))
        self.json_file_directory = os.path.normpath(os.path.dirname(data))
        
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
            self.output_directory = os.path.normpath(self.json_file_directory)
        else:
            self.output_directory = os.path.normpath(os.path.join(self.json_file_directory, loaded_file.pop('output_directory')))

        if loaded_file.get('weight_csv_sensitivity_file') is None:
            raise ValueError('weight_csv_sensitivity_file has not been found in the JSON file')
        self.weight_csv_sensitivity_file = os.path.normpath(os.path.join(self.json_file_directory, loaded_file.pop('weight_csv_sensitivity_file')))

    def factor_data_preperation(self, factors):
        keys = []
        names = []
        files = []
        profiles = []
        raster_sets = []
        for f in factors:
            keys.append(("weight_"+f["name"]))
            names.append(f["name"])
            files.append(os.path.normpath(os.path.join(self.json_file_directory, f["file"])))
            with rasterio.open(os.path.normpath(os.path.join(self.json_file_directory, f["file"]))) as src:
                profiles.append(src.profile)
                band1 = src.read(1)
                masked_data = np.ma.masked_where(band1 == src.nodata, band1)
                raster_sets.append(masked_data)
        return keys, names, files, profiles, raster_sets

    def susceptibility_classes_data_preperation(self, susceptibility_classes):
        names = []
        class_upper_bounds = []
        class_lower_bounds = []
        for s in susceptibility_classes:
            names.append(s["name"])
            class_upper_bounds.append(s["class_upper_bound"])
            class_lower_bounds.append(s["class_lower_bound"])
        return names, class_lower_bounds, class_upper_bounds

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

    def generate_random_weights(self, input_pd_table, data_length):
        random_weights = []
        random_ix = random.randint(0, (data_length-1))
        for nx, name in enumerate(self.factor_names):
            random_weights.append(input_pd_table.at[random_ix,name])
        return random_weights

    def load_weight_csv_sensitivity_file(self, sens_csv):
        csv_file = pd.read_csv(sens_csv)
        csv_file.index.name = 'id'
        number_of_rows = len(csv_file.index)
        return csv_file, number_of_rows

    def overlay_factors(self, factor_weights):
        print("Overlaying factors...")
        for rx, raster in enumerate(self.raster_sets):
            if rx==0:
                overlayed = raster * factor_weights[rx]
            else:
                overlayed += raster * factor_weights[rx]
        susceptibility_class_pixels = self.number_of_pixels_in_susceptibility_classes(susceptibility_raster=overlayed)
        return susceptibility_class_pixels

    def generate_layer(self, factor_weights, suffex):
        print("Generating layer by overlaying factors...")
        sum_max_weight = 0
        for rx, raster in enumerate(self.raster_sets):
            sum_max_weight += np.max(raster)
            if rx==0:
                overlayed = raster * factor_weights[rx]
            else:
                overlayed += raster * factor_weights[rx]
        profile = self.profiles[0]
        with rasterio.open((self.output_directory+"/susceptibility_"+str(suffex)+".tif"), 'w', **profile) as dst:
            dst.write(overlayed, 1)

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
        self.profiles = factor_data[3]
        self.raster_sets = factor_data[4]

        susceptibility_classes_data = self.susceptibility_classes_data_preperation(susceptibility_classes=self.susceptibility_classes)
        self.susceptibility_classes_names = susceptibility_classes_data[0]
        self.susceptibility_classes_lower_bounds = susceptibility_classes_data[1]
        self.susceptibility_classes_upper_bounds = susceptibility_classes_data[2]

        self.check_class_upper_and_lower_bounds(upper=self.susceptibility_classes_upper_bounds,lower=self.susceptibility_classes_lower_bounds)

        #create results dic
        final_results_dic = self.create_results_dict()
        weight_csv_sensitivity_data = self.load_weight_csv_sensitivity_file(self.weight_csv_sensitivity_file)
        sens_table_pd = weight_csv_sensitivity_data[0]
        availabel_items = weight_csv_sensitivity_data[1]
        if (availabel_items<self.trials):
            print("Warning: the number of trials is set to", availabel_items, ", as this is the maximum number of data available in", self.weight_csv_sensitivity_file)

        successful_trials = 0
        while successful_trials < self.trials:
            t0 = time.time()
            weight_inputs = self.generate_random_weights(sens_table_pd, availabel_items)
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

    def setup(self):
        print("Setting up SensitivityEstimator...")
        self.load_data_from_json()

    def generate(self, index, csv_sensitivity):
        print("generating road susceptibility raster layer...")
        csv_file = pd.read_csv(os.path.join(self.output_directory, csv_sensitivity)).set_index('id')
        
        factor_data = self.factor_data_preperation(factors=self.factors)
        self.factor_weight_keys = factor_data[0]
        self.factor_files = factor_data[2]
        self.profiles = factor_data[3]
        self.raster_sets = factor_data[4]

        weight_inputs = []
        for w, ww in enumerate(self.factor_weight_keys):
            weight_inputs.append(csv_file.at[index, ww])
        
        self.generate_layer(factor_weights=weight_inputs, suffex=index)

        print("Layer generated successfully.")

    def compare(self, layer1, layer2):
        print("comparing",layer1,"and",layer2,"...")

        with rasterio.open(os.path.join(self.output_directory,layer1)) as rf:
            ds = rf.read(1)
            lss_np1 = np.ma.masked_where(ds == rf.nodata, ds)
            profile = rf.profile
    
        with rasterio.open(os.path.join(self.output_directory,layer2)) as rf:
            ds = rf.read(1)
            lss_np2 = np.ma.masked_where(ds == rf.nodata, ds)

        susceptibility_classes_data = self.susceptibility_classes_data_preperation(susceptibility_classes=self.susceptibility_classes)
        self.susceptibility_classes_names = susceptibility_classes_data[0]
        self.susceptibility_classes_lower_bounds = susceptibility_classes_data[1]
        self.susceptibility_classes_upper_bounds = susceptibility_classes_data[2]

        self.check_class_upper_and_lower_bounds(upper=self.susceptibility_classes_upper_bounds,lower=self.susceptibility_classes_lower_bounds)

        susceptibility_class_pixels1 = self.number_of_pixels_in_susceptibility_classes(susceptibility_raster=lss_np1)
        susceptibility_class_pixels2 = self.number_of_pixels_in_susceptibility_classes(susceptibility_raster=lss_np2)

        print("Calculating", layer1, "minus", layer2,"...")
        difference_np = np.subtract(lss_np1.astype('int16'), lss_np2.astype('int16'))
        profile['dtype'] = 'int16'
        with rasterio.open(os.path.join(self.output_directory,"susceptibility_difference.tif"), 'w', **profile) as dst:
            dst.write(difference_np, 1)

        print("")
        print("layer1--------------------")
        for s, scn in enumerate(self.susceptibility_classes_names):
            print(scn,":", round(susceptibility_class_pixels1[s],3))

        print("")
        print("layer2--------------------")
        for s, scn in enumerate(self.susceptibility_classes_names):
            print(scn,":", round(susceptibility_class_pixels2[s],3))
