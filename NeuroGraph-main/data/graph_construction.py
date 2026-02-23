import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
from torch_geometric.data import Batch
from utils import *
import re
import argparse
import itertools

# Define default values for datasets, atlases, and construction methods.
DEFAULT_DATASETS = [
    # "ADNI",
    "ADHD",
    # "HCP",
]

DEFAULT_ATLASES = [
    "AAL116",
    # "PP264",
    # "Schaefer100"
]

# The construction_methods dictionary holds mapping names to functions.
DEFAULT_CONSTRUCTION_METHODS = {
    # "cosine_similarity": cosine_similarity,
    "pearson_correlation": pearson_correlation,
    # "partial_correlation": partial_correlation,
    # "correlations_correlation": correlations_correlation,
    # "associated_high_order_fc": associated_high_order_fc,
    # "euclidean_distance": euclidean_distance,
    # "knn_graph": knn_graph,
    # "spearman_correlation": spearman_correlation,
    # "kendall_correlation": kendall_correlation,
    # "mutual_information": mutual_information,
    # "cross_correlation": cross_correlation,
    # "granger_causality": granger_causality,
    # "generalised_synchronisation_matrix": generalised_synchronisation_matrix, # very slow
    # "patels_conditional_dependence_measures_kappa": patels_conditional_dependence_measures_kappa,
    # "patels_conditional_dependence_measures_tau": patels_conditional_dependence_measures_tau,
    # "lingam": lingam, # very slow
}

# Base directories for input data and storing graph results.
INPUT_DATA_DIR = "./fMRIROItimeseries"
BASE_RESULTS_DIR = "./graphconstructionedge"

def get_output_path(base_dir, graph_type, dataset, atlas, method, subject_id, task="rest", hierarchy="dataset_first"):
    """
    Returns the output folder and filename based on the specified hierarchy.
    The filename follows your original format.

    Parameters:
        base_dir (str): The root folder for results.
        graph_type (str): "static" or "dynamic".
        dataset (str): Dataset name.
        atlas (str): Atlas name.
        method (str): Graph construction method.
        subject_id (str): Subject identifier.
        task (str): Task name.
        hierarchy (str): Currently only "dataset_first" is implemented.
        
    Returns:
        output_folder (str), output_filename (str)
    """
    # We use a dataset-first hierarchy: base_dir/graph_type/dataset/atlas/method/
    output_folder = os.path.join(base_dir, graph_type, dataset, atlas, method)
    
    if graph_type == "static":
        filename = f"dataset-{dataset}_sub-{subject_id}_task-{task}_desc-staticgraphconstructionedge_atlas-{atlas}_contrmethd-{method}.pkl"
    else:  # dynamic graphs
        filename = f"dataset-{dataset}_sub-{subject_id}_task-{task}_desc-dynamicgraphconstructionedge_atlas-{atlas}_contrmethd-{method}.pkl"
    
    return output_folder, filename

def process_file(dataset, atlas, method, subject_id, task_name="rest"):
    """
    Processes a single file to generate a static graph.
    """
    try:
        # Construct input file path using INPUT_DATA_DIR
        input_folder = os.path.join(INPUT_DATA_DIR, dataset, atlas)
        input_filepath = os.path.join(
            input_folder,
            f"dataset-{dataset}_sub-{subject_id}_task-{task_name}_desc-fMRIROItimeseries_atlas-{atlas}.pkl"
        )

        # Ensure input folder exists; if not, create it and return a message.
        if not os.path.exists(input_folder):
            os.makedirs(input_folder, exist_ok=True)
            return f"Input folder {input_folder} did not exist; created folder, but file {input_filepath} still missing."
        if not os.path.exists(input_filepath):
            return f"Input file {input_filepath} does not exist."

        # Load data from the input file
        data = pd.read_pickle(input_filepath)

        # Convert data to DataFrame if it is a numpy array
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Construct output path and create directory if needed
        output_folder, output_filename = get_output_path(BASE_RESULTS_DIR, "static", dataset, atlas, method, subject_id, task=task_name)
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(output_folder, output_filename)

        # Skip if output file already exists
        if os.path.exists(output_filepath):
            return f"Output file {output_filepath} already exists. Skipping."

        # Generate the correlation matrix using the specified method
        try:
            correlation_matrix = DEFAULT_CONSTRUCTION_METHODS[method](data)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        except KeyError:
            return f"Method {method} not implemented."

        # Convert the correlation matrix to graph data and save it
        data_list = [correlation_matrix_to_graph_data(correlation_matrix)]
        data_batch = Batch.from_data_list(data_list)
        torch.save(data_batch, output_filepath)

        return f"Processed {output_filepath} successfully."
    except Exception as e:
        return f"Error processing dataset {dataset}, atlas {atlas}, subject {subject_id}, method {method}: {e}"

def process_dynamic_file(dataset, atlas, method, subject_id, task_name="rest"):
    """
    Processes a single file to generate a dynamic graph using sliding windows.
    """
    try:
        # Construct input file path using INPUT_DATA_DIR
        input_folder = os.path.join(INPUT_DATA_DIR, dataset, atlas)
        input_filepath = os.path.join(
            input_folder,
            f"dataset-{dataset}_sub-{subject_id}_task-{task_name}_desc-fMRIROItimeseries_atlas-{atlas}.pkl"
        )

        # Ensure input folder exists; if not, create it and return a message.
        if not os.path.exists(input_folder):
            os.makedirs(input_folder, exist_ok=True)
            return f"Input folder {input_folder} did not exist; created folder, but file {input_filepath} still missing."
        if not os.path.exists(input_filepath):
            return f"Input file {input_filepath} does not exist."

        # Load data from the input file
        data = pd.read_pickle(input_filepath)

        # Convert data to DataFrame if it is a numpy array
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Construct output path and create directory if needed
        output_folder, output_filename = get_output_path(BASE_RESULTS_DIR, "dynamic", dataset, atlas, method, subject_id, task=task_name)
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(output_folder, output_filename)

        # Skip if output file already exists
        if os.path.exists(output_filepath):
            return f"Output file {output_filepath} already exists. Skipping."

        # Generate dynamic correlation matrices using sliding window
        dynamic_correlation_matrixes = []
        time_window = 50  # Window size
        stride = 3        # Sliding stride

        for i in range(0, data.shape[0] - time_window, stride):
            window_data = data.iloc[i:i + time_window]
            correlation_matrix = DEFAULT_CONSTRUCTION_METHODS[method](window_data)
            dynamic_correlation_matrixes.append(correlation_matrix)

        # Convert correlation matrices to graph data and save them
        data_list = [correlation_matrix_to_graph_data(cm) for cm in dynamic_correlation_matrixes]
        data_batch = Batch.from_data_list(data_list)
        torch.save(data_batch, output_filepath)

        return f"Processed {output_filepath} successfully."
    except Exception as e:
        return f"Error processing dataset {dataset}, atlas {atlas}, subject {subject_id}, method {method}: {e}"

def collect_tasks(datasets, atlases, methods):
    """
    Collects processing tasks for all combinations of dataset, atlas, and method.
    Each task is a tuple (dataset, atlas, method, subject_id).
    """
    tasks = []
    for dataset, atlas in itertools.product(datasets, atlases):
        directory = os.path.join(INPUT_DATA_DIR, dataset, atlas)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            tqdm.write(f"Created missing input directory: {directory}")

        # Regex to extract subject IDs from filenames
        pattern = re.compile(
            fr"dataset-{dataset}_sub-(\d+)_task-rest_desc-fMRIROItimeseries_atlas-{atlas}\.pkl"
        )

        subject_ids = [
            pattern.search(filename).group(1)
            for filename in os.listdir(directory)
            if pattern.search(filename)
        ]

        # Append tasks for each method and subject
        for method in methods:
            for subject_id in subject_ids:
                tasks.append((dataset, atlas, method, subject_id))
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Process fMRI timeseries data to construct graphs.")
    parser.add_argument('--mode', type=str, default="default", choices=["default", "finetune"],
                        help="Run mode of the script.")
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS, help="List of datasets to process.")
    parser.add_argument('--atlases', nargs='+', default=DEFAULT_ATLASES, help="List of atlases to process.")
    parser.add_argument('--methods', nargs='+', default=list(DEFAULT_CONSTRUCTION_METHODS.keys()),
                        help="List of construction methods to use.")

    args = parser.parse_args()

    if args.mode == "finetune":
        datasets = args.datasets
        atlases = args.atlases
        methods = args.methods
        print(f"Running in finetune mode with datasets: {datasets}, atlases: {atlases}, methods: {methods}")
    else:
        datasets = DEFAULT_DATASETS
        atlases = DEFAULT_ATLASES
        methods = list(DEFAULT_CONSTRUCTION_METHODS.keys())

    print('datasets:', datasets)
    print('atlases:', atlases)
    print('methods:', methods)

    # Collect tasks to process based on the provided parameters
    tasks = collect_tasks(datasets, atlases, methods)
    
    if len(tasks) == 0:
        print("No tasks found. Please check your input directories and filenames.")
        return

    # Process static graphs
    print("Starting static graph processing...")
    with ProcessPoolExecutor() as executor:
        with tqdm(total=len(tasks), desc="Static Graph Progress", unit="task") as pbar:
            future_to_task = {
                executor.submit(
                    process_file,
                    dataset,
                    atlas,
                    method,
                    subject_id
                ): (dataset, atlas, method, subject_id)
                for dataset, atlas, method, subject_id in tasks
            }
            for future in as_completed(future_to_task):
                dataset, atlas, method, subject_id = future_to_task[future]
                try:
                    status = future.result()
                    tqdm.write(status)
                except Exception as exc:
                    tqdm.write(f"Error in {dataset}, {atlas}, {method}, {subject_id}: {exc}")
                finally:
                    pbar.update(1)

    print("Static graph processing complete.")

    # # Process dynamic graphs
    # print("Starting dynamic graph processing...")
    # with ProcessPoolExecutor() as executor:
    #     with tqdm(total=len(tasks), desc="Dynamic Graph Progress", unit="task") as pbar:
    #         future_to_task = {
    #             executor.submit(
    #                 process_dynamic_file,
    #                 dataset,
    #                 atlas,
    #                 method,
    #                 subject_id
    #             ): (dataset, atlas, method, subject_id)
    #             for dataset, atlas, method, subject_id in tasks
    #         }
    #         for future in as_completed(future_to_task):
    #             dataset, atlas, method, subject_id = future_to_task[future]
    #             try:
    #                 status = future.result()
    #                 tqdm.write(status)
    #             except Exception as exc:
    #                 tqdm.write(f"Error in {dataset}, {atlas}, {method}, {subject_id}: {exc}")
    #             finally:
    #                 pbar.update(1)

    # print("Dynamic graph processing complete.")

if __name__ == "__main__":
    main()
