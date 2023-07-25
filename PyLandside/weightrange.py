#loading neccessery packages
import os
import click
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from osgeo import gdal
import random
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error, confusion_matrix
logger = logging.getLogger(__name__)


class BaseModel(object):
    def __init__(self, json_file, csvfile_name = None, sol_index = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_file = json_file
        self.csvfile_name = csvfile_name
        self.sol_index = sol_index
        self.objectives = None
        self.zero_constraints = None
        self.objective_values = {}
        self.zero_constraints_values = {}
        self.variables = []
        self.variable_upper_bounds = []
        self.variable_lower_bounds = []
        self.variables_values = {}
        self.original_variables_values = {}
        self.cut_fill_limit = None

    def load_model_json_file(self, **kwargs):
        """Load a model from a file
        """
        data = os.path.join(os.getcwd(), self.json_file)
        #data = self.json_file
        if isinstance(data, str):
            # argument is a filename
            logging.info('Loading model from file: "{}"'.format(data))
            with open(data, "r") as f:
                loaded_file = json.load(f)

        if loaded_file.get('dem_dir') is None:
            raise ValueError('dem_dir  has not been found in the input JSON file')
        
        self.dem_file = os.path.join(os.getcwd(), loaded_file.pop('dem_dir'))
        self.progress_reporting_interval = loaded_file.pop('progress_reporting_interval')
        self.objectives = loaded_file.pop('objectives')

        try:
            self.zero_constraints = loaded_file.pop('zero_constraints')
        except Exception:
            pass

        self.cut_fill_limit = loaded_file.pop('cut_fill_limit')
        self.compaction_factor = loaded_file.pop('compaction_factor')
        self.hv_calc = loaded_file.pop('hv_calc')

        return loaded_file


    def feature_importance_model(target, features, max_tree_depth, n_estimators, cores):
        #create a RandomForestClassifier 
        model = RandomForestClassifier(random_state=1, max_depth=max_tree_depth, n_estimators = n_estimators, n_jobs = cores)
        model.fit(features,target)
        #calculate relative importance
        importances = model.feature_importances_
        return importances, model
        

    def overall_accuracy(mod,X_train,Y_train,X_test,Y_test):

        Y_predicted_test = mod.predict(X_test)
        Y_predicted_train = mod.predict(X_train)

        matrix_test = confusion_matrix(Y_test, Y_predicted_test)
        matrix_train = confusion_matrix(Y_train, Y_predicted_train)

        #when binary
        tn_test, fp_test, fn_test, tp_test = matrix_test.ravel()
        tn_train, fp_train, fn_train, tp_train = matrix_train.ravel()

        #weight of 


        TP_TN_test = 0
        FP_fn_test = 0
        for y in range(len(matrix_test)):
            for x in range(len(matrix_test)):
                if y == x:
                    TP_TN_test += matrix_test[y][x]
                else:
                    FP_fn_test += matrix_test[y][x]


        TP_TN_train = 0
        FP_fn_train = 0
        print(matrix_train)
        for y in range(len(matrix_train)):
            for x in range(len(matrix_train)):
                if y == x:
                    TP_TN_train += matrix_train[y][x]
                else:
                    FP_fn_train += matrix_train[y][x]

        overall_accuracy_test = TP_TN_test/(TP_TN_test+FP_fn_test)
        overall_accuracy_train = TP_TN_train/(TP_TN_train+FP_fn_train)
        
        print('overall accuracy in training =', round(overall_accuracy_train,3))
        print('overall accuracy in testing =', round(overall_accuracy_test,3))

        return overall_accuracy_train, overall_accuracy_test


    def load_data_from_csv(filename, ix):
        logger.info('Loading input file: "{}"'.format(filename))
        temp = pd.read_csv(filename)
        input_data = temp.set_index(ix)
        return input_data


    def generate_feature_importance_sens(targets_file,features_file, depths_file, sensiterations, test_sample_size, n_estimators, cores, perfromance_cutoff=0.75):
        
        targets_df = load_data_from_csv(targets_file, ix = 'id')
        features_df = load_data_from_csv(features_file, ix = 'id')
        tree_depths_df = load_data_from_csv(depths_file, ix = 'target')
        
        #Cach target and feature names
        target_names = targets_df.columns
        feature_names = features_df.columns
        feature_names1 = list(feature_names)
        feature_names1.append("target")
        feature_names1.append("training_overall_accuracy")
        feature_names1.append("testing_overall_accuracy")


        #create an empty df based on the number of targets and features under consideration
        results = pd.DataFrame(index = list(range(int(sensiterations))), columns = feature_names1)
        results.index.name = 'iteration'

        for itera in range(int(sensiterations)):
            #Note: This randomly split the data in 80% train and 20% test data
            X_train, X_test, Y_train, Y_test = train_test_split(features_df, targets_df, test_size = test_sample_size)

            #Loop through targets
            for t, targ in enumerate (target_names):

                print('_____________________________________')
                print('Iteration number:',t+1+(itera*len(target_names)))

                #create a RandomForestClassifier model
                importances = feature_importance_model(target=Y_train[targ], features=X_train, max_tree_depth=tree_depths_df.at[targ,"tree_depth"], n_estimators=n_estimators, cores=cores)

                metrics = overall_accuracy(importances[1],X_train,Y_train[targ],X_test,Y_test[targ])

                #save the values of relative importance into the 'results' dataframe
                if metrics[0] >= perfromance_cutoff and metrics[1] >= perfromance_cutoff:
                    results.at[itera, "target"] = targ
                    results.at[itera, "training_overall_accuracy"] = metrics[0]
                    results.at[itera, "testing_overall_accuracy"] = metrics[1]
                    for f, feat in enumerate (feature_names):
                        results.at[itera, feat] = importances[0][f]

            results.dropna(how='all').to_csv('weight_ranges.csv')


    def load_dataset(filename):
        logger.info('Loading input file: "{}"'.format(filename))
        path = os.path.join(os.getcwd(), filename)
        raster_file = gdal.Open(path)
        return raster_file


    def dem_to_numpy(raster, no_data_value):
        logger.info('Converting input file: "{}" with NoDataValue of "{}"'.format(raster, no_data_value))
        raster_numpy = np.array(raster.GetRasterBand(1).ReadAsArray()).astype(float)
        raster_numpy[raster_numpy==no_data_value]=np.nan
        return raster_numpy


    def plot_raster_from_np_array(np_array):
        plt.figure()
        plt.imshow(np_array)
        plt.show()


    def class_road_cells(risk_raster, road_dist_raster):
        extremly_high_cells = 0
        high_cells = 0
        moderate_cells = 0
        low_cells = 0
        very_low_cells = 0
        for x_cor in range(risk_raster.shape[1]):
            for y_cor in  range(risk_raster.shape[0]):
                if str(risk_raster[y_cor,x_cor])!="nan":
                    if road_dist_raster[y_cor,x_cor]<0.1:
                        if risk_raster[y_cor,x_cor]>6 and risk_raster[y_cor,x_cor]<=9:
                            extremly_high_cells += 1
                        elif risk_raster[y_cor,x_cor]>5 and risk_raster[y_cor,x_cor]<=6:
                            high_cells += 1
                        elif risk_raster[y_cor,x_cor]>4 and risk_raster[y_cor,x_cor]<=5:
                            moderate_cells += 1
                        elif risk_raster[y_cor,x_cor]>2 and risk_raster[y_cor,x_cor]<=4:
                            low_cells += 1
                        elif risk_raster[y_cor,x_cor]>0 and risk_raster[y_cor,x_cor]<=2:
                            very_low_cells += 1
        return extremly_high_cells, high_cells, moderate_cells, low_cells, very_low_cells


    def road_risk_raster_generator(risk_raster, road_dist_raster):
        new_raster = np.zeros_like(risk_raster)
        for x_cor in range(risk_raster.shape[1]):
            for y_cor in  range(risk_raster.shape[0]):
                if str(risk_raster[y_cor,x_cor])!="nan":
                    if road_dist_raster[y_cor,x_cor]<0.1:
                        if risk_raster[y_cor,x_cor]>6 and risk_raster[y_cor,x_cor]<=9:
                            new_raster[y_cor,x_cor] = 5
                        elif risk_raster[y_cor,x_cor]>5 and risk_raster[y_cor,x_cor]<=6:
                            new_raster[y_cor,x_cor] = 4
                        elif risk_raster[y_cor,x_cor]>4 and risk_raster[y_cor,x_cor]<=5:
                            new_raster[y_cor,x_cor] = 3
                        elif risk_raster[y_cor,x_cor]>2 and risk_raster[y_cor,x_cor]<=4:
                            new_raster[y_cor,x_cor] = 2
                        elif risk_raster[y_cor,x_cor]>0 and risk_raster[y_cor,x_cor]<=2:
                            new_raster[y_cor,x_cor] = 1
                    else:
                        new_raster[y_cor,x_cor] = np.nan
                else:
                    new_raster[y_cor,x_cor] = np.nan

        return new_raster


    def raster_from_numpy(numpy_array, projection, geo_trans, output_name="new_dem.tif"):
        driver = gdal.GetDriverByName("GTiff")
        driver.Register()
        outds = driver.Create(output_name, xsize = numpy_array.shape[1],
                            ysize = numpy_array.shape[0], bands = 1, 
                            eType = gdal.GDT_Float64)
        outds.SetGeoTransform(geo_trans)
        outds.SetProjection(projection)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(numpy_array)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()


    def overlay_factors_roads(weights):
        calc_formula = "(A*"+str(weights["weight_land_cover"])+")+(B*"+str(weights["weight_lithology"])+")+(C*"+str(weights["weight_rainfall"])+")+(D*"+str(weights["weight_road_dist"])+")+(E*"+str(weights["weight_slope"])+")"      
        cmd_sum = "gdal_calc.py --quiet --overwrite --extent=union --outfile temp_lss.tif -A land_use/land_use.tif -B lithology/lithology.tif -C rainfall/rainfall_30yr.tif -D road_dist/road_dist.tif -E slope/slope.tif --calc="+calc_formula 
        os.system(cmd_sum)

        lss_ds = load_dataset("temp_lss.tif")
        lss_np = dem_to_numpy(raster=lss_ds, no_data_value=lss_ds.GetRasterBand(1).GetNoDataValue())

        road_dis_stretched_ds = load_dataset("road_dist/road_dist_stretched.tif")
        road_dis_stretched_np = dem_to_numpy(raster=road_dis_stretched_ds, no_data_value=road_dis_stretched_ds.GetRasterBand(1).GetNoDataValue())

        class_roads_cells = class_road_cells(risk_raster=lss_np, road_dist_raster=road_dis_stretched_np)

        return class_roads_cells[0], class_roads_cells[1], class_roads_cells[2], class_roads_cells[3], class_roads_cells[4]




'''
class BaseModel(object):
    def __init__(self, json_file, csvfile_name = None, sol_index = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_file = json_file
        self.csvfile_name = csvfile_name
        self.sol_index = sol_index
        self.objectives = None
        self.zero_constraints = None
        self.objective_values = {}
        self.zero_constraints_values = {}
        self.variables = []
        self.variable_upper_bounds = []
        self.variable_lower_bounds = []
        self.variables_values = {}
        self.original_variables_values = {}
        self.cut_fill_limit = None

    def load_model_json_file(self, **kwargs):
        """Load a model from a file
        """
        data = os.path.join(os.getcwd(), self.json_file)
        #data = self.json_file
        if isinstance(data, str):
            # argument is a filename
            logging.info('Loading model from file: "{}"'.format(data))
            with open(data, "r") as f:
                loaded_file = json.load(f)

        if loaded_file.get('dem_dir') is None:
            raise ValueError('dem_dir  has not been found in the input JSON file')
        
        self.dem_file = os.path.join(os.getcwd(), loaded_file.pop('dem_dir'))
        self.progress_reporting_interval = loaded_file.pop('progress_reporting_interval')
        self.objectives = loaded_file.pop('objectives')

        try:
            self.zero_constraints = loaded_file.pop('zero_constraints')
        except Exception:
            pass

        self.cut_fill_limit = loaded_file.pop('cut_fill_limit')
        self.compaction_factor = loaded_file.pop('compaction_factor')
        self.hv_calc = loaded_file.pop('hv_calc')

        return loaded_file

    def load_dataset(self, filename):
        logger.info('Loading input file: "{}"'.format(filename))
        path = os.path.join(os.getcwd(), filename)
        raster_file = gdal.Open(path)
        self.gt = raster_file.GetGeoTransform()
        self.proj = raster_file.GetProjection()
        return raster_file

    def dem_to_numpy(self, raster, no_data_value):
        logger.info('Converting input file: "{}" with NoDataValue of "{}"'.format(raster, no_data_value))
        raster_numpy = np.array(raster.GetRasterBand(1).ReadAsArray()).astype(float)
        raster_numpy[raster_numpy==no_data_value]=np.nan
        return raster_numpy

    def plot_raster_from_np_array(self, np_array):
        plt.figure()
        plt.imshow(np_array)
        plt.show()


class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.DEM_np = None
        self.slope_margin = 0.01

    def check_cell_sink(self, numpy_raster, x, y):
        array_shape = numpy_raster.shape
        if x==0 or y==0:
            #print("X or Y = 0")
            return 0
        elif x==array_shape[1]-1 or y==array_shape[0]-1:
            #print("X or Y = max values")
            return 0
        elif str(numpy_raster[y-1,x-1])=="nan" or str(numpy_raster[y,x-1])=="nan" or str(numpy_raster[y+1,x-1])=="nan" or str(numpy_raster[y-1,x])=="nan" or str(numpy_raster[y+1,x])=="nan" or str(numpy_raster[y-1,x+1])=="nan" or str(numpy_raster[y,x+1])=="nan" or str(numpy_raster[y+1,x+1])=="nan":
            #print("X or Y are next to NAN")
            return 0
        elif numpy_raster[y,x] <= numpy_raster[y-1,x-1] and numpy_raster[y,x] <= numpy_raster[y,x-1] and numpy_raster[y,x] <= numpy_raster[y+1,x-1] and numpy_raster[y,x] <= numpy_raster[y-1,x] and numpy_raster[y,x] <= numpy_raster[y+1,x] and numpy_raster[y,x] <= numpy_raster[y-1,x+1] and numpy_raster[y,x] <= numpy_raster[y,x+1] and numpy_raster[y,x] <= numpy_raster[y+1,x+1]:
            return 1
        else:
            return 0

    def save_variable_names(self):
        from .optimization.moea import MOEAVariable
        for variable_data in self.variable_preperation(numpy_raster = self.DEM_np):
            self.variables.append(str(variable_data[0])) 
            self.original_variables_values[str(variable_data[0])] = variable_data[1]    

    def non_nan_cells(self, numpy_raster, x, y):
        if str(numpy_raster[y,x])=="nan":
            pass
        else:
            var_name  = "var"+str(x)+"_"+str(y)
            return var_name

    def variable_preperation(self, numpy_raster):
        for x_cor in range(numpy_raster.shape[1]):
            for y_cor in  range(numpy_raster.shape[0]):
                if str(numpy_raster[y_cor,x_cor])!="nan":
                    yield self.non_nan_cells(numpy_raster,x_cor,y_cor), numpy_raster[y_cor,x_cor]

    def get_objectives(self):
        from .optimization.moea import MOEAObjective
        for objective in list(self.objectives):
            yield MOEAObjective(objective, minimise=True, epsilon=1)

    def get_objective_values(self):
        for objective in list(self.objectives):
            yield self.objective_values[objective]

    def get_constraints(self):
        from .optimization.moea import MOEAConstraint
        if self.zero_constraints is not None:
            for constraint in list(self.zero_constraints):
                yield MOEAConstraint(name=constraint, operator="==", value=0)

    def get_constraint_values(self):
        if self.zero_constraints is not None:
            for constraint in list(self.zero_constraints):
                yield self.zero_constraints_values[constraint]


    def get_variables(self):
        from .optimization.moea import MOEAVariable
        self.save_variable_names
        for variable in self.variables:
            upper_bound = self.original_variables_values[variable] + self.cut_fill_limit
            lower_bound = self.original_variables_values[variable] - self.cut_fill_limit
            self.variable_upper_bounds.append(upper_bound)
            self.variable_lower_bounds.append(lower_bound)
            yield MOEAVariable(str(variable), lower_bounds=round(lower_bound,2),upper_bounds=round(upper_bound,2))              

    def apply_variables(self, solution):
        i=0
        for variable in list(self.variables):
            self.variables_values[variable]=solution[i]
            i += 1

    def from_variables_to_numpy(self, initial_numpy_raster):
        i=0
        output_array = copy.deepcopy(initial_numpy_raster)
        for x_cor in range(initial_numpy_raster.shape[1]):
            for y_cor in  range(initial_numpy_raster.shape[0]):
                if str(initial_numpy_raster[y_cor,x_cor])!="nan":
                    output_array[y_cor,x_cor] = self.variables_values[self.variables[i]]
                    i+=1
                else:
                    pass
        return output_array

    def sink_counter(self, numpy_raster, progress_interval=15):
        self.sinks = np.zeros_like(numpy_raster)
        temp = 1
        for x_cor in range(numpy_raster.shape[1]):
            for y_cor in  range(numpy_raster.shape[0]):
                self.sinks[y_cor,x_cor] = self.check_cell_sink(numpy_raster,x_cor,y_cor)
                percent = (x_cor*numpy_raster.shape[0]+y_cor)/(numpy_raster.shape[1] *numpy_raster.shape[0])*100

        return self.sinks.sum(axis=1).sum(axis=0)

    def cut_and_fill(self, dem_org, dem_mod):
        self.cut_fill_values = np.zeros_like(dem_org)
        self.cut_volume = 0
        self.fill_volume = 0
        for x_cor in range(dem_org.shape[1]):
            for y_cor in  range(dem_org.shape[0]):
                if str(dem_mod[y_cor,x_cor]) == "nan" or str(dem_org[y_cor,x_cor]) == "nan":
                    pass
                else:
                    self.cut_fill_values[y_cor,x_cor] = dem_mod[y_cor,x_cor] - dem_org[y_cor,x_cor]
                    if self.cut_fill_values[y_cor,x_cor] < 0:
                        self.cut_volume += self.cut_fill_values[y_cor,x_cor]
                    elif self.cut_fill_values[y_cor,x_cor] > 0:
                        self.fill_volume += self.cut_fill_values[y_cor,x_cor]

        return self.cut_volume, self.fill_volume


    def cut_fill_credit(self, fill_credit, dem_np):
        current_fill_credit = fill_credit
        output_array = copy.deepcopy(dem_np)
        for x_cor in range(dem_np.shape[1]):
            for y_cor in  range(dem_np.shape[0]):
                if str(dem_np[y_cor,x_cor]) == "nan":
                    pass
                else:
                    cut_fill_data = self.calc_fill_cut(output_array, x=x_cor, y=y_cor)
                    fill_depth = min([cut_fill_data[0],current_fill_credit])
                    cut_depth = cut_fill_data[1]

                    current_fill_credit -= fill_depth

                    output_array[y_cor,x_cor] = output_array[y_cor,x_cor] + fill_depth + cut_depth

        return output_array, current_fill_credit


    def calc_fill_cut(self, numpy_raster, x, y):
        array_shape = numpy_raster.shape
        if x==0 or y==0:
            #print("X or Y = 0")
            return 0, 0
        elif x==array_shape[1]-1 or y==array_shape[0]-1:
            #print("X or Y = max values")
            return 0, 0
        elif str(numpy_raster[y-1,x-1])=="nan" or str(numpy_raster[y,x-1])=="nan" or str(numpy_raster[y+1,x-1])=="nan" or str(numpy_raster[y-1,x])=="nan" or str(numpy_raster[y+1,x])=="nan" or str(numpy_raster[y-1,x+1])=="nan" or str(numpy_raster[y,x+1])=="nan" or str(numpy_raster[y+1,x+1])=="nan":
            #print("X or Y are next to NAN")
            return 0, 0
        elif numpy_raster[y,x] <= numpy_raster[y-1,x-1] and numpy_raster[y,x] <= numpy_raster[y,x-1] and numpy_raster[y,x] <= numpy_raster[y+1,x-1] and numpy_raster[y,x] <= numpy_raster[y-1,x] and numpy_raster[y,x] <= numpy_raster[y+1,x] and numpy_raster[y,x] <= numpy_raster[y-1,x+1] and numpy_raster[y,x] <= numpy_raster[y,x+1] and numpy_raster[y,x] <= numpy_raster[y+1,x+1]:
            min_val = min([numpy_raster[y-1,x-1], numpy_raster[y,x-1], numpy_raster[y+1,x-1], numpy_raster[y-1,x], numpy_raster[y+1,x], numpy_raster[y-1,x+1], numpy_raster[y,x+1], numpy_raster[y+1,x+1]])
            fill_val = min_val - numpy_raster[y,x] + self.slope_margin
            return fill_val, 0
        else:
            min_val = min([numpy_raster[y-1,x-1], numpy_raster[y,x-1], numpy_raster[y+1,x-1], numpy_raster[y-1,x], numpy_raster[y+1,x], numpy_raster[y-1,x+1], numpy_raster[y,x+1], numpy_raster[y+1,x+1]])
            cut_val = min_val - numpy_raster[y,x] + self.slope_margin
            return 0, cut_val


    def path_and_elev_difference(self, dem_np, xx, yy):
        x_in = xx
        y_in = yy
        flow_path_raster = copy.deepcopy(dem_np)
        point_level = dem_np[y_in,x_in]
        exit_level = None

        flow_path_x = []
        flow_path_y = []

        while exit_level is None:
            flow_path_data = self.flow_path(flow_path_raster, x_in, y_in)
            x_in = flow_path_data[0]
            y_in = flow_path_data[1]
            flow_path_raster = flow_path_data[2]
            exit_level = flow_path_data[3]
            if x_in is not None and y_in is not None:
                flow_path_x.append(x_in)
                flow_path_y.append(y_in)




    def flow_path(self, flow_path, x, y):
        if str(flow_path[y-1,x-1])=="nan" or str(flow_path[y,x-1])=="nan" or str(flow_path[y+1,x-1])=="nan" or str(flow_path[y-1,x])=="nan" or str(flow_path[y+1,x])=="nan" or str(flow_path[y-1,x+1])=="nan" or str(flow_path[y,x+1])=="nan" or str(flow_path[y+1,x+1])=="nan":
            return None, None, flow_path, flow_path[y,x]
        else:
            min_value_around = min([flow_path[y-1,x-1], flow_path[y,x-1], flow_path[y+1,x-1], flow_path[y-1,x], flow_path[y+1,x], flow_path[y-1,x+1], flow_path[y,x+1], flow_path[y+1,x+1]])
            if flow_path[y-1,x-1] == min_value_around:
                x_val = x-1
                y_val = y-1
                flow_path[y-1,x-1]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y,x-1] == min_value_around:
                x_val = x-1
                y_val = y
                flow_path[y,x-1]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y+1,x-1] == min_value_around:
                x_val = x-1
                y_val = y+1
                flow_path[y+1,x-1]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y-1,x] == min_value_around:
                x_val = x
                y_val = y-1
                flow_path[y-1,x]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y+1,x] == min_value_around:
                x_val = x
                y_val = y+1
                flow_path[y+1,x]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y-1,x+1] == min_value_around:
                x_val = x+1
                y_val = y-1
                flow_path[y-1,x+1]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y,x+1] == min_value_around:
                x_val = x+1
                y_val = y
                flow_path[y,x+1]=9999999999
                return x_val, y_val, flow_path, None
            elif flow_path[y+1,x+1] == min_value_around:
                x_val = x+1
                y_val = y+1
                flow_path[y+1,x+1]=9999999999
                return x_val, y_val, flow_path, None

    def setup(self):
        self.load_model_json_file()
        logger.info('Setting up model based on the file: "{}"'.format(self.dem_file))
        self.dataset = self.load_dataset(self.dem_file)
        self.DEM_np = self.dem_to_numpy(raster=self.dataset, no_data_value=self.dataset.GetRasterBand(1).GetNoDataValue())

    def run(self):
        print('Running model...')
        self.credit = 0
        zero_credit_counter = 0
        for iteration in range(0,100000000):
            if iteration == 0:
                self.new_dem, self.credit = self.cut_fill_credit(fill_credit=self.credit, dem_np=self.DEM_np)
            else:
                self.new_dem, self.credit = self.cut_fill_credit(fill_credit=self.credit, dem_np=self.new_dem)

            a = self.sink_counter(numpy_raster=self.new_dem)
            b = self.sink_counter(numpy_raster=self.DEM_np)
            
            print("iteration number", iteration)
            print("sinks new and original", a, b, self.credit)

            if self.credit == 0:
                zero_credit_counter += 1

            if zero_credit_counter >= 3:
                self.credit += 1000000
                zero_credit_counter = 0

            if a == 0:
                break

        driver = gdal.GetDriverByName("GTiff")
        driver.Register()
        outds = driver.Create("new_dem.tif", xsize = self.new_dem.shape[1],
                            ysize = self.new_dem.shape[0], bands = 1, 
                            eType = gdal.GDT_Float64)
        outds.SetGeoTransform(self.gt)
        outds.SetProjection(self.proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(self.new_dem)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()


    def create_dem(self):
        moea_data = pd.read_csv(os.path.join(os.getcwd(), self.csvfile_name))
        temp_i = 0
        variables_results = []
        self.new_dem = np.zeros_like(self.DEM_np)

        for v, variable in enumerate(moea_data.columns):
            if variable.startswith("var"):
                variables_results.append(variable)

        #looping through variables
        for x_cor in range(self.DEM_np.shape[1]):
            for y_cor in  range(self.DEM_np.shape[0]):
                if str(self.DEM_np[y_cor,x_cor])!="nan":
                    design_value = moea_data[variables_results[temp_i]].iloc[self.sol_index]
                    self.new_dem[y_cor,x_cor] = design_value
                    temp_i += 1
                else:
                    self.new_dem[y_cor,x_cor] = np.nan

 
        driver = gdal.GetDriverByName("GTiff")
        driver.Register()
        outds = driver.Create("new_dem.tif", xsize = self.new_dem.shape[1],
                            ysize = self.new_dem.shape[0], bands = 1, 
                            eType = gdal.GDT_Float64)
        outds.SetGeoTransform(self.gt)
        outds.SetProjection(self.proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(self.new_dem)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()

        a = self.sink_counter(numpy_raster=self.new_dem)
        b = self.sink_counter(numpy_raster=self.DEM_np)
        
        print("sinks new and original", a, b)
'''
                            