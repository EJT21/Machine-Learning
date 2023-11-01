#Importing Libraries
import os
import shutil
import pandas as pd
import sys
# Initialize an empty DataFrame of the variables you want
columns = ["Variable1", "Variable2", "Variable3"]
data_frame = pd.DataFrame(columns=columns)
# Specify file paths
data_file_path = "your_file.txt"
destination_directory = "your_chosen_directory"
#------------------------------------------------------------------------------------------------------------------#
def read_data_file(data_file_path):
    #Logic for walking through the data file
    try: 
        with open(data_file_path, "r") as file:
            data_file_path = "your_file.txt" #Use the walk method here
    except FileNotFoundError as file_error:
        print(f"File not found: {file_error}")
    except PermissionError as permission_error:
        print(f"Permission denied when trying to access the file: {permission_error}")

def run_report(data_frame):
# Open the file and process its lines
    with open(data_file_path, "r") as file:
        # Variables you are looking for
        Variable1 = ""
        Variable2 = ""
        Variable3 = ""
        for line in file:
            #method is used to remove leading and trailing whitespace
            line = line.strip()
            #Get each word in each line
            words = line.split()
            if "Variable1" in line:
                Variable1 = line.split("Variable 1: ")[1] #Grab value after this keyword
            elif "Variabl2" in line:
                Variable2 = words[5] #Grab the 5th word in that line
            elif "Variable3" in line:
                Variable3 = line #Grab the whole line
            #Once you have all the variables, put it in a dataframe as a single row    
            if Variable1 and Variable2 and Variable3:
                data = {
                    "Variable1": Variable1,
                    "Variable2": Variable2,
                    "Variable3": Variable3
                    }
                data_frame = data_frame.append(data, ignore_index=True)
                Variable1 = ""
                Variable2 = ""
                Variable3 = ""

def export_to_csv(data_frame, csv_data_file_path):
    data_frame.to_csv(csv_data_file_path, index=False)


def move_file(source_path, destination_directory):
    if os.path.exists(source_path) and os.path.exists(destination_directory):
        shutil.move(source_path, destination_directory)
        print(f"File moved to {destination_directory}")

# Step 1: Read the data and see if there are any errors, if there is, terminate the program
read_data_file(data_file_path)

# Step 2: Run the report
run_report(data_frame)

# Step 3: Export the data to a CSV file
export_to_csv(data_frame, csv_data_file_path)

# Step 4: Move the data file to a chosen directory and to the archieve folder
move_file(data_data_file_path, destination_directory)