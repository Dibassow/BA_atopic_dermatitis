import anndata as ad
import scanpy as sc
from typing import List
import numpy as np
from scipy.stats import median_abs_deviation
import seaborn as sns
import pandas as pd

def load_mtx(mtx_path: str, barcodes_path: str, features_path: str, sample_name: str | None = None) -> ad.AnnData:
    adata = sc.read_mtx(mtx_path).transpose()
    
    barcodes = pd.read_csv(barcodes_path, header=None, sep='\t', names=['barcodes'])
    features = pd.read_csv(features_path, header=None, sep='\t', names=['gene_ids', 'gene_names'])

    adata.obs_names = sample_name + "_" + barcodes['barcodes'] if sample_name is not None else barcodes['barcodes']
    adata.var_names = features['gene_ids']
    adata.var['gene_names'] = features['gene_names'].values
    
    if sample_name is not None:
        adata.obs['sample'] = sample_name

    return adata

def __is_outlier__(adata: ad.AnnData, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


def __annotate_genes__(adata: ad.AnnData, use_col: str | None = None):
    col = adata.var_names if use_col is None else adata.var[use_col]

    adata.var["mt"] = col.str.startswith("MT-")
    adata.var["ribo"] = col.str.startswith(("RPL", "RPS"))
    adata.var["hb"] = col.str.contains(("^HB[^(P)]"))


def calculate_qc_metrics(adata: ad.AnnData, percent_top: List[int] = [20], use_col: str | None = None):
    __annotate_genes__(adata, use_col=use_col)

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True,
        percent_top=percent_top,
    )


def perform_qc_filtering(
    adata: ad.AnnData,
    mads_log_total_counts: int = 5,
    mads_log_n_genes_by_counts: int = 5,
    limit_pct_counts_mt: int = 8,
) -> ad.AnnData:
    adata.obs["outlier"] = (
        __is_outlier__(adata, "log1p_total_counts", mads_log_total_counts)
        | __is_outlier__(adata, "log1p_n_genes_by_counts", mads_log_n_genes_by_counts)
    )

    print(f"Number of outliers: {adata.obs['outlier'].sum()}")

    adata.obs["mt_outlier"] = (
        adata.obs["pct_counts_mt"] > limit_pct_counts_mt
    )

    print(f"Number of mt outliers: {adata.obs['mt_outlier'].sum()}")

    filtered = adata[~adata.obs["outlier"] & ~adata.obs["mt_outlier"]].copy()

    return filtered

def plot_qc_metrics(adata: ad.AnnData):
    sns.displot(adata.obs["total_counts"], bins=100, kde=False)
    sc.pl.violin(adata, "pct_counts_mt")
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

def perform_qc_filtering_lower(
    adata: ad.AnnData,
    mads_log_total_counts: int = 5,
    mads_log_n_genes_by_counts: int = 5,
    limit_pct_counts_mt: int = 8,
) -> ad.AnnData:
    adata.obs["outlier"] = (
        __is_outlier__(adata, "log1p_total_counts", mads_log_total_counts)
        | __is_outlier__(adata, "log1p_n_genes_by_counts", mads_log_n_genes_by_counts)
    )

    print(f"Number of outliers: {adata.obs['outlier'].sum()}")

    adata.obs["mt_outlier"] = (
        adata.obs["pct_counts_mt"] < limit_pct_counts_mt
    )

    print(f"Number of mt outliers: {adata.obs['mt_outlier'].sum()}")

    filtered = adata[~adata.obs["outlier"] & ~adata.obs["mt_outlier"]].copy()

    return filtered