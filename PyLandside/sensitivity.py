#loading neccessery packages
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import logging
import random
import copy
from datetime import datetime
logger = logging.getLogger(__name__)

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
                            