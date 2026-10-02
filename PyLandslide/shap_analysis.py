#loading necessary packages
import os
import json
import pandas as pd
import logging
import joblib
import numpy as np
import shap
import matplotlib

#to save plots without opening a GUI
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class SHAPAnalysis(object):
    """
    This class  includes methods for the SHAP analysis of produced 
    machine learning models for landslide susceptibility
    """

    def __init__(self, json_file, *args, **kwargs):
        """
        Initialise a new SHAPAnalysis object.

        Args:
            json_file: JSON-based document specifying the configuration information for performing SHAP analysis
            calculation.
        """
        super().__init__(*args, **kwargs)
        self.json_file = json_file

    def setup(self):
        print("Setting up SHAP analysis")
        self.load_data_from_json()

    def load_data_from_json(self):
        """
        Loads the configuration JSON-based document assigned to self.json_file. Extracts the data from the JSON-based
        document and assign them to self.model_number, self.model_directory, self.features_file, self.output_directory,
        self.sample_size, self. max_displayed, and self.random_state
        """
        data = os.path.normpath(os.path.join(os.getcwd(), self.json_file))
        self.json_file_directory = os.path.normpath(os.path.dirname(data))

        with open(data,"r") as f: #opening json file for SHAP analysis settings
            loaded_file= json.load(f)

        # the original iteration number from the weight range csv
        self.model_number = loaded_file["model_number"]

        # path to the folder that contains the saved models
        self.model_directory = os.path.normpath(os.path.join(self.json_file_directory,loaded_file["models_directory"]))

        #Set the path to the original features csv
        self.features_file = os.path.join(self.json_file_directory,loaded_file["features_file"])

        #Add a folder for SHAP outputs
        self.output_directory = os.path.normpath(os.path.join(self.json_file_directory, loaded_file["output_directory"]))

        #the number of feature rows to exam
        self.sample_size = loaded_file.get("sample_size",500)

        # the number of factors displayed in the graph
        self.max_displayed = loaded_file.get("max_displayed",20)

        self.random_state = loaded_file.get("random_state",42)


    def load_model (self):
        """
        Loads the machine learning models for the SHAP Analysis
        """

        model_filename = (f"rf_model_iteration_{self.model_number + 1}.joblib")

        model_path = os.path.join(self.model_directory,model_filename)
        
        # add a error if the specified file is not found
        if not  os.path.isfile(model_path):
            raise FileNotFoundError(f"Saved model not found: {model_path}")

        print("Loading model: ",model_path)

        return joblib.load(model_path)

    def load_features(self,model):

        """
        Loads the features or landslide factors from the existing features.csv file, returns features_df
        """

        #get the features df
        features_df = pd.read_csv(self.features_file).set_index("id")

        #sample the rows to keep the computation time managable

        if self.sample_size is not None:

            if self.sample_size <= 0:
                raise ValueError("sample size must be a positive interger or null")

            if len(features_df) > self.sample_size:
                features_df= features_df.sample(n=self.sample_size,random_state=self.random_state)

        return features_df

    def calculate_shap_values(self,model,features_df):
        """
        Calculates shap values for a given model based on given features, returns shap_values
        """

        print("calculating SHAP values for ",len(features_df),"feature rows")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(features_df)

        #classifier is binary (0/1 for landslide or non landslide of specified type), we want to select the postivite class (1)
        if shap_values.values.ndim == 3: # checks if the shap values hae 3 dimensions, number of locations, number of predictor factors, number of model classes
            positive_class_index = list(model.classes_).index(1)
            shap_values = shap_values[:,:,positive_class_index]

        return shap_values

    def save_shap_data(self,shap_values,features_df):
        """
        Produces a summary csv file of the shap values
        """

        #get the SHAP value for each factor at each sample location
        shap_df = pd.DataFrame(shap_values.values,index=features_df.index, columns = features_df.columns)

        #change the index column name to "id"
        shap_df.index.name = "id"

        # create the output directory for the csv
        shap_csv = os.path.join(self.output_directory,f"shape_values_iteration_{self.model_number}.csv")

        #save the features df for the original sampled locations
        features_csv_shap = os.path.join(self.output_directory,f"shape_features_iteration_{self.model_number}.csv")
        features_df.to_csv(features_csv_shap,index_label="id")

        # save the shap_df and features df to a csv file
        shap_df.to_csv(shap_csv)

        #create a summary df for the statists of the SHAP analysis
        summary_df = pd.DataFrame({"factor":features_df.columns,"mean_absolute_shap":np.abs(shap_values.values).mean(axis=0)})

        summary_df = summary_df.sort_values("mean_absolute_shap",ascending=False)

        summary_csv = os.path.join(self.output_directory,f"shap_summary_iteration{self.model_number}.csv")

        summary_df.to_csv(summary_csv,index = False)

        #Save the complete SHAP file
        full_shap_file = os.path.join(self.output_directory,f"full_shap_file_{self.model_number}.joblib")
        joblib.dump(shap_values,full_shap_file)

        print("SHAP values saved to:",shap_csv)
        print("Sampled features daved to:",features_csv_shap)
        print("SHAP summary saved to:",summary_csv)
        print("Full SHAP file saved to:", full_shap_file)



    def execute(self):
        """
        execute the shap calculation and saves a beeswarm plot and the SHAP value csv for the selected iteration
        """

        os.makedirs(self.output_directory,exist_ok = True)

        model = self.load_model()

        features_df = self.load_features(model)

        shap_values = self.calculate_shap_values(model,features_df)
        self.save_shap_data(shap_values,features_df)

        plt.figure()

        shap.plots.beeswarm(shap_values,max_display= self.max_displayed,show = False)

        plt.title (f"SHAP analysis - model iteration {self.model_number}")

        plt.tight_layout ()

        output_path = os.path.join(self.output_directory,f"shap_beeswarm_iteration_{self.model_number}.png")

        plt.savefig(output_path ,dpi = 300, bbox_inches = "tight")

        plt.close()
        print("SHAP plot saved to: ",output_path)





            










