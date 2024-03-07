#loading neccessery packages
import os
import json
import pandas as pd
import numpy as np
import logging
import geopandas as gpd
from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio

class DataPreparation(object):
    """
    This is a data preparation class that includes methods for co-registering raster layers and extracting
    point data for Machine Learning.
    """

    def __init__(self, json_file=None, folder_name=None, *args, **kwargs):
        """
        Initialise a new DataPreparation object.

        Args:
            json_file: JSON-based document specifying the configuration information for performing data preparation.

            folder_name: Folder containing the raster data.
        """
        super().__init__(*args, **kwargs)
        self.json_file = json_file
        self.folder_name = folder_name

    def load_data_from_json(self):
        """
        Loads the configuration JSON-based document assigned to self.json_file. Extracts the data from the JSON-based
        document and assign them to self.factors, self.output_directory, self.landslide_locations, and
        self.nonlandslide_locations.

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

        if loaded_file.get('output_directory') is None:
            print('output directory set to the cwd. If you wish to change this default value, add output_directory to the inputs provided in the JSON file')
            self.output_directory = os.path.normpath(self.json_file_directory)
        else:
            self.output_directory = os.path.normpath(os.path.join(self.json_file_directory, loaded_file.pop('output_directory')))

        if loaded_file.get('landslide_locations') is None:
            raise ValueError('landslide_locations has not been found in the JSON file')
        self.landslide_locations = os.path.normpath(os.path.join(self.json_file_directory, loaded_file.pop('landslide_locations')))

        if loaded_file.get('nonlandslide_locations') is None:
            raise ValueError('nonlandslide_locations has not been found in the JSON file')
        self.nonlandslide_locations = os.path.normpath(os.path.join(self.json_file_directory, loaded_file.pop('nonlandslide_locations')))

    def setup(self):
        """
        Calls the load_data_from_json method to extract the information provided in the JSON-based document.
        """
        self.load_data_from_json()
        print('Setting up WeightRangePreparation based on the file: "{}"'.format(self.json_file))

    def factor_data_preperation(self, factors):
        """
        Takes a list of dictionaries that include factor names and their associated raster files and returns
        lists of factor names, datasets, and datasets as arrays.
        """
        names = []
        sets = []
        sets_arrays = []
        for f in factors:
            names.append(f["name"])
            set_temp = rasterio.open(os.path.normpath(os.path.join(self.json_file_directory, f["file"])))
            sets.append(set_temp)
            sets_arrays.append(set_temp.read(1))
        return names, sets, sets_arrays

    def extract(self):
        """
        Extracts the factor values at the landslide and non-landslide locations. These locations are obtained from the
        shapefiles provided in the JSON-based document. The results are saved into features.csv (factor values) and
        targets.csv (landslide or non-landslide status) in the output directory specified in the JSON-based document.
        """
        print("Preparing and extracting data...")
        factor_data = self.factor_data_preperation(self.factors)
        self.factor_names = factor_data[0]
        self.factor_sets = factor_data[1]
        self.factor_sets_arrays = factor_data[2]

        self.landslide_locations_shp = gpd.read_file(self.landslide_locations)
        self.nonlandslide_locations_shp = gpd.read_file(self.nonlandslide_locations)

        features = self.create_results_dict(self.factor_names)
        targets = self.create_results_dict(['status_id'])

        for nx, name in enumerate(self.factor_names):
            temp_index = 0
            for px, point in enumerate(self.landslide_locations_shp['geometry']):
                x=point.xy[0][0]
                y=point.xy[1][0]
                row, col = self.factor_sets[nx].index(x,y)
                extracted_value = self.factor_sets_arrays[nx][row,col]
                if extracted_value==self.factor_sets[nx].nodata:
                    features[name].append(np.nan)
                    if nx==0:
                        targets['status_id'].append(np.nan)
                    else:
                        targets['status_id'][temp_index] = np.nan
                else:
                    features[name].append(extracted_value)
                    if nx==0:
                        targets['status_id'].append(1)
                temp_index += 1

            for px, point in enumerate(self.nonlandslide_locations_shp['geometry']):
                x=point.xy[0][0]
                y=point.xy[1][0]
                row, col = self.factor_sets[nx].index(x,y)
                extracted_value = self.factor_sets_arrays[nx][row,col]
                if extracted_value==self.factor_sets[nx].nodata:
                    features[name].append(np.nan)
                    if nx==0:
                        targets['status_id'].append(np.nan)
                    else:                    
                        targets['status_id'][temp_index] = np.nan
                else:
                    features[name].append(extracted_value)
                    if nx==0: 
                        targets['status_id'].append(0)
                temp_index += 1

        #write the results
        features_df = pd.DataFrame.from_dict(features)
        features_df1 = features_df.dropna(how='any').reset_index(drop=True)
        features_df1.index.name = 'id'
        features_df1.to_csv(os.path.join(self.output_directory,'features.csv'))
        
        targets_df = pd.DataFrame.from_dict(targets)
        targets_df1 = targets_df.dropna(how='any').reset_index(drop=True)
        targets_df1.index.name = 'id'
        targets_df1.to_csv(os.path.join(self.output_directory,'targets.csv'))

        print("Completed.")

    def create_results_dict(self, index_array):
        """
        Creates a dictionary for saving the results of the extract() method. This method takes the names indices (in
        this case factor names) as an argument.
        """
        results_dic = {}
        for n in index_array:
            results_dic[n]=[]
        return results_dic

    def adjust(self):
        """
        Looks into the self.folder_name variable and pre-process the raster files located in it by converting them to uint8
        to reduce data size. The adjusted files are saved into "folder_name/uint8".
        """
        dir_list = os.listdir(os.path.join(os.getcwd(), self.folder_name))
        raster_files = []
        outfiles = []
        for f in dir_list:
            if f.endswith('.tif') or f.endswith('.tiff'):
                raster_files.append(os.path.join(os.getcwd(), self.folder_name, f))
                outfiles.append(os.path.join(os.getcwd(), self.folder_name, "uint8", f))

        if not os.path.exists(os.path.join(os.getcwd(), self.folder_name, "uint8")):
            os.makedirs(os.path.join(os.getcwd(), self.folder_name, "uint8"))

        for rx, rff in enumerate(raster_files):
            with rasterio.open(rff) as rf:
                ds = rf.read(1)
                masked_ds = np.ma.masked_where(ds == rf.nodata, ds)
                profile = rf.profile
                profile['nodata'] = 255
                profile['dtype'] = 'uint8'
                print(rff)
                with rasterio.open(outfiles[rx], 'w', **profile) as dst:
                    dst.write(masked_ds, 1)

    def align(self):
        """
        Co-registers the rasters adjusted with the adjust() method by looking into "folder_name/uint8". The
        co-registration is performed to ensure that all rasters are aligned, have the same resolution, and same array
        sizes. The resulting co-registered rasters are saved into "folder_name/alinged_rasters".
        """
        dir_list = os.listdir(os.path.join(os.getcwd(), self.folder_name))
        raster_files = []
        outfiles = []
        for f in dir_list:
            if f.endswith('.tif') or f.endswith('.tiff'):
                raster_files.append(os.path.join(os.getcwd(), self.folder_name, "uint8", f))
                outfiles.append(os.path.join(os.getcwd(), self.folder_name, "alinged_rasters", f))

        if not os.path.exists(os.path.join(os.getcwd(), self.folder_name, "alinged_rasters")):
            os.makedirs(os.path.join(os.getcwd(), self.folder_name, "alinged_rasters"))

        raster_width = []
        raster_height = []
        bound_left = []
        bound_bottom = []
        bound_right = []
        bound_top = []
        coordinate_system = []
        for rff in raster_files:
            with rasterio.open(rff) as rf:
                raster_width.append(rf.width)
                raster_height.append(rf.height)
                bound_left.append(rf.bounds[0])
                bound_bottom.append(rf.bounds[1])
                bound_right.append(rf.bounds[2])
                bound_top.append(rf.bounds[3])
                coordinate_system.append(rf.crs)

        dst_transform = []
        dst_width = []
        dst_height = []
        for ix, rd in enumerate(raster_files):
            dst_transform_temp, dst_width_temp, dst_height_temp = calculate_default_transform(
                src_crs=coordinate_system[ix],
                dst_crs=coordinate_system[0], 
                width=min(raster_width),
                height=min(raster_height),
                left=max(bound_left),
                bottom=max(bound_bottom),
                right=min(bound_right),
                top=min(bound_top)
            )
            dst_transform.append(dst_transform_temp)
            dst_width.append(dst_width_temp)
            dst_height.append(dst_height_temp)

        for ix, rff in enumerate(raster_files):
            with rasterio.open(rff) as rf: 
                ddd = rf.read(1)
                dst_kwargs = rf.meta.copy()
                dst_kwargs.update({"crs": coordinate_system[0],
                                "transform": dst_transform[ix],
                                "width": dst_width[ix],
                                "height": dst_height[ix],
                                "dtype": 'uint8',
                                "nodata": rf.nodata,
                                'compress': 'lzw'})

                with rasterio.open(outfiles[ix], "w", **dst_kwargs) as dst:
                    print(outfiles[ix])
                    for i in range(1, rf.count + 1):
                        reproject(
                            source=rasterio.band(rf, i),
                            destination=rasterio.band(dst, i),
                            src_transform=rf.transform,
                            src_crs=rf.crs,
                            dst_transform=dst_transform[0],
                            dst_crs=coordinate_system[0],
                            num_threads=4,
                            resampling=Resampling.nearest)
