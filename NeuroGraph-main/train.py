import os
import json
import torch
import datetime
import time
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from multiprocessing import Pool, cpu_count, set_start_method
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")  # suppress all warnings

from utils import * 
from model import *


def train(model, args: Args, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(args.device)
        # print('data:', data)
        out = model(data) 
        loss = criterion(out, data.y) 
        total_loss +=loss
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss/len(train_loader.dataset)


@torch.no_grad()
def test(model, args: Args, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            out = model(data)
            probs = F.softmax(out, dim=1)  # Calculate probabilities
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy()[:, 1])  # Keep the probabilities of the positive class
            all_labels.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    metrics = {
        'accuracy': accuracy,
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1
    }
    
    return metrics


def train_and_evaluate_fold(args, train_loader, val_loader, test_loader, fold_idx, verbose=False):
    """
    Train and evaluate model for a single fold
    
    Args:
        args: Arguments object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        fold_idx: Fold index
        verbose: Whether to print detailed information
    
    Returns:
        dict: Test metrics for this fold
    """
    
    checkpoints_dir = './checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    # Initialize model
    gnn = eval(args.model)
    model = ResidualGNNs(args, train_loader.dataset, args.hidden, args.hidden_mlp, args.num_layers, gnn).to(args.device)
    
    if verbose:
        print(f"Model: {model}")
    
    # Training loop
    best_val_auroc = float("-inf")
    val_acc_history, test_acc_history, test_loss_history = [], [], []
    train_loss_history = []
    val_auroc_history = []
    patience = getattr(args, "early_stop_patience", None)
    min_delta = getattr(args, "early_stop_min_delta", 0.0) or 0.0
    patience_counter = 0
    epochs_ran = 0
    
    model_prefix = os.path.join(
        checkpoints_dir,
        f"{args.dataset}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}_fold{fold_idx+1}"
    )
    best_model_path = f"{model_prefix}_best.pkl"

    for epoch in tqdm(range(args.epochs), desc=f"Fold {fold_idx + 1}", position=args.rank + 1, leave=True):
        epochs_ran = epoch + 1
        # Train
        train_loss = train(model, args, train_loader)
        train_loss_value = float(train_loss.item() if torch.is_tensor(train_loss) else train_loss)
        train_loss_history.append(train_loss_value)
        
        # Validate
        val_metrics = test(model, args, val_loader)
        val_auroc_value = float(val_metrics['auroc'])
        val_auroc_history.append(val_auroc_value)
        
        if verbose:
            train_metrics = test(model, args, train_loader)
            test_metrics = test(model, args, test_loader)
            print(f"Epoch {epoch}: Loss={train_loss_value:.6f}, Val_AUROC={val_metrics['auroc']:.4f}, Test_AUROC={test_metrics['auroc']:.4f}")
        
        # Save best model based on validation AUROC
        if val_metrics['auroc'] > best_val_auroc + min_delta:
            best_val_auroc = val_metrics['auroc']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            if patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[EARLY STOP] Fold {fold_idx + 1} stopped at epoch {epoch + 1}")
                    break
    
    # Load best model and evaluate on test set
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model checkpoint not found for fold {fold_idx + 1}: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_metrics = test(model, args, test_loader)


    # Comment out below to skip saving/plotting curves
    _save_training_history(
        model_prefix=model_prefix,
        epochs=list(range(1, len(train_loss_history) + 1)),
        train_losses=train_loss_history,
        val_aurocs=val_auroc_history
    )
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return test_metrics, epochs_ran


def _save_training_history(model_prefix, epochs, train_losses, val_aurocs):
    """Save and plot training loss and validation AUROC curves."""
    length = min(len(epochs), len(train_losses), len(val_aurocs))
    if length == 0:
        return
    epochs = epochs[:length]
    train_losses = train_losses[:length]
    val_aurocs = val_aurocs[:length]

    history_path = f"{model_prefix}_history.csv"
    try:
        with open(history_path, "w") as f:
            f.write("epoch,train_loss,val_auroc\n")
            for e, tl, va in zip(epochs, train_losses, val_aurocs):
                f.write(f"{e},{tl:.6f},{va:.6f}\n")
    except OSError as e:
        print(f"[WARNING] Failed to write history CSV ({history_path}): {e}")

    plot_path = f"{model_prefix}_training_curve.png"
    try:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(epochs, train_losses, color="tab:blue", label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(epochs, val_aurocs, color="tab:orange", label="Val AUROC")
        ax2.set_ylabel("Val AUROC", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0.0, 1.0)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines + lines2, labels + labels2, loc="upper center", ncol=2)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Failed to plot training curve ({plot_path}): {e}")


def _fold_progress_file(args):
    checkpoints_dir = './checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    resume_filename = f"{args.dataset}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}_fold_progress.json"
    return os.path.join(checkpoints_dir, resume_filename)


def _load_fold_progress(args):
    progress = {"completed_folds": {}, "elapsed_seconds": 0.0, "fold_epochs": {}}
    path = _fold_progress_file(args)
    if not os.path.exists(path):
        return progress
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            completed = data.get("completed_folds")
            if isinstance(completed, dict):
                progress["completed_folds"] = {
                    k: v for k, v in completed.items() if isinstance(v, dict)
                }
            elapsed = data.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                progress["elapsed_seconds"] = float(elapsed)
            fold_epochs = data.get("fold_epochs")
            if isinstance(fold_epochs, dict):
                progress["fold_epochs"] = {
                    k: int(v) for k, v in fold_epochs.items() if isinstance(v, (int, float))
                }
    except (OSError, json.JSONDecodeError):
        pass
    return progress


def _save_fold_progress(args, progress):
    path = _fold_progress_file(args)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f)
    os.replace(tmp_path, path)


def _sanitize_metrics(metrics):
    return {key: float(val) for key, val in metrics.items()}


def predefined_5fold_cross_validation(args, verbose=False):
    """
    Perform 5-fold cross validation using predefined fold assignments
    
    Args:
        args: Arguments object
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (avg_metrics, std_metrics, total_runtime_seconds)
    """
    
    # Define the 5 fold combinations (fixed rotation: 3 train, 1 val, 1 test)
    fold_combinations = [
        {'train_folds': [1, 2, 3], 'val_fold': 4, 'test_fold': 5},
        {'train_folds': [2, 3, 4], 'val_fold': 5, 'test_fold': 1},
        {'train_folds': [3, 4, 5], 'val_fold': 1, 'test_fold': 2},
        {'train_folds': [4, 5, 1], 'val_fold': 2, 'test_fold': 3},
        {'train_folds': [5, 1, 2], 'val_fold': 3, 'test_fold': 4},
    ]
    
    progress = _load_fold_progress(args)
    completed_folds = progress.get("completed_folds", {})
    stored_fold_epochs = progress.get("fold_epochs", {}) or {}
    fold_epoch_counts = dict(stored_fold_epochs)
    total_runtime = float(progress.get("elapsed_seconds", 0.0) or 0.0)
    fold_metrics = []
    fold_failed = False
    epoch_order = {}
    
    for fold_idx, fold_info in enumerate(fold_combinations):
        fold_key = str(fold_idx + 1)
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/5: Train={fold_info['train_folds']}, Val={fold_info['val_fold']}, Test={fold_info['test_fold']}")
        print(f"{'='*60}")
        if fold_key in completed_folds:
            metrics = _sanitize_metrics(completed_folds[fold_key])
            completed_folds[fold_key] = metrics
            fold_metrics.append(metrics)
            if fold_key in fold_epoch_counts:
                epoch_order[fold_key] = fold_epoch_counts[fold_key]
            print(f"[INFO] Fold {fold_idx + 1} already completed. Skipping training.")
            continue
        
        fold_start = time.perf_counter()
        try:
            # Load data for this fold combination
            train_data, val_data, test_data, train_labels, val_labels, test_labels = load_data_with_fold_split(args, fold_info)
            
            if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
                print(f"[WARNING] Empty dataset in fold {fold_idx + 1}. Skipping.")
                total_runtime += time.perf_counter() - fold_start
                progress["elapsed_seconds"] = total_runtime
                _save_fold_progress(args, progress)
                continue
            
            # Create data loaders
            train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_data, args.batch_size, shuffle=False, drop_last=False)
            test_loader = DataLoader(test_data, args.batch_size, shuffle=False, drop_last=False)
            
            # Train model for this fold (function moved to train.py)
            fold_metrics_result, epochs_ran = train_and_evaluate_fold(args, train_loader, val_loader, test_loader, fold_idx, verbose)
            sanitized_metrics = _sanitize_metrics(fold_metrics_result)
            fold_metrics.append(sanitized_metrics)
            completed_folds[fold_key] = sanitized_metrics
            progress["completed_folds"] = completed_folds
            fold_epoch_counts[fold_key] = epochs_ran
            epoch_order[fold_key] = epochs_ran
            progress["fold_epochs"] = fold_epoch_counts
            total_runtime += time.perf_counter() - fold_start
            progress["elapsed_seconds"] = total_runtime
            _save_fold_progress(args, progress)
            
            # Clean up memory
            del train_data, val_data, test_data, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            
        except Exception as e:
            total_runtime += time.perf_counter() - fold_start
            progress["elapsed_seconds"] = total_runtime
            _save_fold_progress(args, progress)
            fold_failed = True
            print(f"[ERROR] Failed to process fold {fold_idx + 1}: {e}")
            continue
    
    if not fold_metrics:
        raise ValueError("No valid folds were processed!")
    
    pending_folds = [idx for idx in range(1, len(fold_combinations) + 1) if str(idx) not in completed_folds]
    if fold_failed and pending_folds:
        raise RuntimeError(f"Cross validation interrupted before completing folds {pending_folds}. Resolve the error and rerun to resume.")
    
    ordered_metrics = [completed_folds[str(idx)] for idx in range(1, len(fold_combinations) + 1) if str(idx) in completed_folds]
    
    # Calculate average metrics
    avg_metrics = {key: np.mean([fold[key] for fold in ordered_metrics]) for key in ordered_metrics[0].keys()}
    std_metrics = {key: np.std([fold[key] for fold in ordered_metrics]) for key in ordered_metrics[0].keys()}
    
    if not pending_folds:
        resume_path = _fold_progress_file(args)
        if os.path.exists(resume_path):
            try:
                os.remove(resume_path)
            except OSError:
                pass
    
    if verbose:
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Average Metrics: {avg_metrics}")
        print(f"Std Metrics: {std_metrics}")
    
    args.total_runtime_seconds = total_runtime
    args.total_runtime_readable = str(datetime.timedelta(seconds=int(total_runtime)))
    # Attach epoch statistics to args for logging
    ordered_epoch_counts = {}
    for idx in range(1, len(fold_combinations) + 1):
        key = str(idx)
        ordered_epoch_counts[key] = epoch_order.get(key) or fold_epoch_counts.get(key)
    epoch_values = [v for v in ordered_epoch_counts.values() if isinstance(v, (int, float))]
    args.fold_epoch_counts = ordered_epoch_counts
    if epoch_values:
        args.avg_epochs_run = float(np.mean(epoch_values))
    else:
        args.avg_epochs_run = None
    return avg_metrics, std_metrics, total_runtime, ordered_metrics


if_method_has_constructed = []
def bench_from_args(args: Args, verbose = False, use_predefined_folds=False):

    experiment_start = time.perf_counter()
    method = args.dataset + args.atlas + args.edge_dir_prefix
    # construct the dataset
    if method in if_method_has_constructed:
        print(f"Graph construction for {args.dataset} with {args.edge_dir_prefix} already completed.")
    else:
        # For our ADHD data, graphs are pre-built from roi_matrices_775.npy;
        # if timeseries dir is missing, skip graph construction and reuse existing graph files.
        try:
            construct_graphs_for_all_subjects(
                args.dataset,
                args.atlas,
                args.edge_dir_prefix,
                args.rank,
                input_data_dir=args.timeseries_dir,
            )
        except FileNotFoundError as e:
            print(f"[WARNING] Skip graph construction for {args.dataset}-{args.atlas}-{args.edge_dir_prefix}: {e}")
        if_method_has_constructed.append(method)

    # Record time spent before cross validation begins (e.g., graph construction)
    pre_cv_elapsed = time.perf_counter() - experiment_start

    # Use predefined folds if requested
    if use_predefined_folds:
        print("[INFO] Using predefined 5-fold cross validation")
        progress = _load_fold_progress(args)
        progress["elapsed_seconds"] = float(progress.get("elapsed_seconds", 0.0) or 0.0) + pre_cv_elapsed
        _save_fold_progress(args, progress)

        avg_metrics, std_metrics, total_runtime, fold_metrics = predefined_5fold_cross_validation(args, verbose)
        args.total_runtime_seconds = total_runtime
        args.total_runtime_readable = str(datetime.timedelta(seconds=int(total_runtime)))
        log_experiment_result(args, avg_metrics, std_metrics, fold_metrics, "predefined_folds")
        return avg_metrics, std_metrics

    # Original random fold logic
    full_data = load_data_from_args(args)

    # Initialize KFold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_data)):
        print(f"Fold {fold + 1}/{args.n_splits}")

        # Create train and validation data loaders for this fold
        train_data = [full_data[i] for i in train_idx]
        test_data = [full_data[i] for i in test_idx]

        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=args.seed)

        # create data loaders
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=False, drop_last=False)

        checkpoints_dir = './checkpoints/'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        val_acc_history, test_acc_history, test_loss_history = [],[],[]
        #seed = 42
        for index in range(args.runs):
            gnn = eval(args.model)
            model = ResidualGNNs(args, train_data, args.hidden, args.hidden_mlp, args.num_layers, gnn).to(args.device) ## apply GNN*
            if (verbose):
                print(model)
            #total_params = sum(p.numel() for p in model.parameters())
            loss, test_acc = [], []
            best_val_auroc = float("-inf")
            patience = getattr(args, "early_stop_patience", None)
            min_delta = getattr(args, "early_stop_min_delta", 0.0) or 0.0
            patience_counter = 0
            for epoch in tqdm(range(args.epochs), desc=f"{args.tune_name}", position=args.rank + 1, leave=True):
                loss = train(model, args, train_loader)
                val_metrics = test(model, args, val_loader)
                
                if verbose:
                    train_metrics = test(model, args, train_loader)
                    test_metrics = test(model, args, test_loader)
                    print("epoch: {}, loss: {}, \ntrain_metrics:{}, \nval_metrics:{}, \ntest_metrics:{}".format(
                        epoch, np.round(loss.item(), 6), train_metrics, val_metrics, test_metrics))
                
                # Check if this is the best validation AUROC
                if val_metrics['auroc'] > best_val_auroc + min_delta:
                    best_val_auroc = val_metrics['auroc']
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{checkpoints_dir}{args.dataset}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}task-checkpoint-best-auroc.pkl")
                else:
                    if patience is not None:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"[EARLY STOP] {args.tune_name} stopped at epoch {epoch + 1}")
                            break
        #test the model
        model_path = f"{checkpoints_dir}{args.dataset}_{args.edge_dir_prefix.split('/')[0]}_{args.model}{args.tune_name}task-checkpoint-best-auroc.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model checkpoint not found: {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_metrics = test(model, args, test_loader)
        fold_metrics.append(test_metrics)
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        if (verbose):
            print(f"Fold {fold + 1} Test Metrics: {test_metrics}")

        if (verbose):
            print('test_metrics:', test_metrics)
    # Aggregate results
    avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0].keys()}

    # get results std of each metric on all folds
    std_metrics = {key: np.std([fold[key] for fold in fold_metrics]) for key in fold_metrics[0].keys()}
    
    if getattr(args, "total_runtime_seconds", None) is None:
        args.total_runtime_seconds = time.perf_counter() - experiment_start
        args.total_runtime_readable = str(datetime.timedelta(seconds=int(args.total_runtime_seconds)))
    
    if verbose:
        print(f"Average Metrics: {avg_metrics}")
        print(f"Std Metrics: {std_metrics}")
    
    
    log_experiment_result(args, avg_metrics, std_metrics, fold_metrics)
    return avg_metrics, std_metrics


# Configuration: True = predefined 5-fold cross-validation, False = random 5-fold split
USE_PREDEFINED_FOLDS = True  # Change to False for random folds

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # absolute path of this file
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # parent of BASE_DIR

argsDictTune_a = {
    # dataset: ADHD (this repo); NeuroGraph also supports HCP etc.
    'dataset' : ["ADHD"],
    'dataset_dir' : os.path.join(BASE_DIR, "data"),
    # Use in-project ROI timeseries path (not hardcoded external path).
    # Expected layout: data/fMRIROItimeseries/ADHD/{atlas}/dataset-ADHD_sub-*_desc-fMRIROItimeseries_atlas-{atlas}.pkl
    'timeseries_dir': os.path.join(BASE_DIR, "data", "fMRIROItimeseries"),
    'folds_dir': PROJECT_ROOT,
    # choose from: GCNConv, GINConv, SGConv, GeneralConv, GATConv
    'edge_dir_prefix' : [
        'pearson_correlation',
        #'cosine_similarity',
        #'euclidean_distance',
        #'spearman_correlation',
        #'kendall_correlation',
        #'partial_correlation',
        #'cross_correlation',
        #'correlations_correlation',
        #'associated_high_order_fc',
        #'knn_graph',
        # 'mutual_information',
        # 'granger_causality',
        # 'coherence_matrix',
        # 'generalised_synchronisation_matrix',
        #'patels_conditional_dependence_measures_kappa',
        #'patels_conditional_dependence_measures_tau',
    ],
    # Our ADHD data uses AAL116 (116x116 ROI Pearson matrix) only
    'atlas' : ["AAL116"],
    'model' : "GCNConv" ,
    'num_classes' : 2,
    # Fixed config for quick pipeline verification; change back to lists for large-scale search.
    'weight_decay' : 5e-4,
    'batch_size': 32,
    'hidden_mlp' : 64,
    'hidden': 32,
    'num_layers' : 2,
    'runs' : 1,
    'lr': 5e-4,
    'epochs' : 100,
    'edge_percent_to_keep': 0.3,
    'n_splits' : 5,
    'seed' : 42,
    'graph_type' : 'static',
    'early_stop_patience': 60,
    'early_stop_min_delta': 1e-4,
    # 'verbose' : True
}

args_list_a = Args.tuning_list(argsDictTune_a)
fix_seed(args_list_a[0].seed)

#restart
existing_result_keys = set()
result_file = f"result_{args_list_a[0].dataset}.txt"

if os.path.exists(result_file):
    with open(result_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("_dataset"):
                existing_result_keys.add(line)


args_list_a = [a for a in args_list_a if a.tune_name not in existing_result_keys]

gpu_count = torch.cuda.device_count()
available_gpu_ids = list(range(gpu_count)) if gpu_count > 0 else []

# Assign a round-robin GPU and tqdm rank per task
for i, args in enumerate(args_list_a):
    if len(available_gpu_ids) > 0:
        args.gpu_id = available_gpu_ids[i % len(available_gpu_ids)]
        args.device = f"cuda:{args.gpu_id}"
    else:
        args.gpu_id = None
        args.device = "cpu"
    args.rank = i

# Single-task execution function
def run_single_experiment(args: Args, use_predefined_folds=False):
    if torch.cuda.is_available() and (args.gpu_id is not None):
        torch.cuda.set_device(args.gpu_id)
    print(f"[INFO] Running {args.tune_name} on {args.device}")
    if use_predefined_folds:
        print(f"[INFO] Using predefined folds for {args.tune_name}")
    fix_seed(args.seed)
    met, std = bench_from_args(args, verbose=False, use_predefined_folds=use_predefined_folds)
    runtime_seconds = getattr(args, "total_runtime_seconds", None)
    runtime_readable = getattr(args, "total_runtime_readable", None)
    with open(f"result_{args.dataset}.txt", "a") as f:
        f.write(f"{args.tune_name}\n{met}\n")
        if runtime_seconds is not None:
            f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
            if runtime_readable is not None:
                f.write(f"runtime_readable: {runtime_readable}\n")
        f.write("--------------------------------\n")
    if runtime_seconds is not None:
        display_runtime = runtime_readable if runtime_readable else f"{runtime_seconds:.2f}s"
        print(f"[INFO] Runtime for {args.tune_name}: {display_runtime} ({runtime_seconds:.2f}s)")
    print(f"✅ {args.tune_name} done on {args.device}")
    return met

# Max number of parallel tasks (can set up to cpu_count() if desired)
max_parallel_tasks = max(1, gpu_count)


# Start task pool
if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    print(f"[INFO] Using {'predefined' if USE_PREDEFINED_FOLDS else 'random'} folds")
    
    with Pool(processes=min(max_parallel_tasks, cpu_count())) as pool:
        # Pass the USE_PREDEFINED_FOLDS flag to each experiment
        test_metric_list_a = pool.starmap(run_single_experiment, [(args, USE_PREDEFINED_FOLDS) for args in args_list_a])
