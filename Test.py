import os
import shutil
import pandas as pd
from nltk.tokenize import word_tokenize
import sys
# Initialize an empty DataFrame of the variables you want
columns = ["Card Number", "Name", "Address"]
data_frame = pd.DataFrame(columns=columns)
# Specify the file path
file_path = "your_file.txt"  # Replace with your file path
# Specify file paths
data_file_path = "your_file.txt"
csv_file_path = "report_data.csv"
destination_directory = "your_chosen_directory"
#------------------------------------------------------------------------------------------------------------------#
def read_data_file(file_path):
    #Logic for walking through the data file
    try: 
        with open(file_path, "r") as file:
    except FileNotFoundError as file_error:
        print(f"File not found: {file_error}")
    except PermissionError as permission_error:
        print(f"Permission denied when trying to access the file: {permission_error}")
    return data_frame

def run_report(data_frame):
    # Open the file and process its lines
    with open(file_path, "r") as file:
    # Variables you are looking for
    card_number = ""
    name = ""
    address = ""
    for line in file:
           try:
                line = line.strip()
                if "Card Number" in line:
                    card_number = line.replace("Card Number: ", "")
                elif line.startswith("Name: "):
                    name = line.replace("Name: ", "")
                elif line.startswith("Address: "):
                    address = line.replace("Address: ", "")

                if card_number and name and address:
                    data = {
                        "Card Number": card_number,
                        "Name": name,
                        "Address": address
                    }
                    data_frame = data_frame.append(data, ignore_index=True)
                    card_number = ""
                    name = ""
                    address = ""
            except Exception as line_error:
                print(f"Error processing a line: {line_error}")
    pass

def export_to_csv(data_frame, csv_file_path):
    data_frame.to_csv(csv_file_path, index=False)


def move_file(source_path, destination_directory):
    if os.path.exists(source_path) and os.path.exists(destination_directory):
        shutil.move(source_path, destination_directory)
        print(f"File moved to {destination_directory}")

# Read the data
data_frame = read_data_file(data_file_path)

# Generate and run the report
run_report(data_frame)

# Export the data to a CSV file
export_to_csv(data_frame, csv_file_path)

# Move the data file to a chosen directory
move_file(data_file_path, destination_directory)
