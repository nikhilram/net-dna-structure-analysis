"""
NET DNA image analysis pipeline

Computes:
- Z vs B structural balance (skeleton-based log_ZB)
- Z-DNA domain organization (nucleation vs elongation)
- Z/B composition (intensity + structure)
- 8-oxoG spatial relationships
- ZBP1 colocalization
- Mitochondrial vs nuclear Z-DNA

Input: multichannel fluorescence images
Output: per-image quantitative metrics

Author: Nikhil Ram Mohan, Ph.D.
"""

from pathlib import Path
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from skimage import io, exposure, filters, morphology, measure
from skimage.filters import sobel, threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, white_tophat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import tifffile as tiff


CHANNELS = {
    "B-DNA": {"index": 0, "type": "dna"},
    "Z-DNA": {"index": 1, "type": "antibody"},
    "ZBP1": {"index": 2, "type": "antibody"},
    "MITO": {"index": 3, "type": "antibody"}
}

treatment_groups = {
    "CQ_DNase1L3": ["cq_1l3"],
    "PMA_DNase1L3": ["pma_1l3"],

    "CQ_DNase1": ["cq_dnase1"],
    "PMA_DNase1": ["pma_dnase1"],

    "CQ": ["_cq"],
    "PMA": ["_pma"],
}



def is_control_file(filename):
    return "control" in filename.lower()

def save_multichannel_visualization(
    img,
    channel_data,
    nuclei_mask,
    exclusion_zone,
    net_regions,
    all_filaments,
    results,
    image_path
):

    # -----------------------------------------
    # Extract channels
    # -----------------------------------------

    b_img = channel_data["B-DNA"].get("quant_image", channel_data["B-DNA"].get("image"))
    b_mask = channel_data["B-DNA"]["mask"]

    z_img = channel_data["Z-DNA"].get("quant_image", channel_data["Z-DNA"].get("image"))
    z_mask = channel_data["Z-DNA"]["mask"]

    # -----------------------------------------
    # Composite overlay
    # -----------------------------------------

    composite = np.zeros((*b_img.shape, 3))

    composite[..., 0] = b_img   # red
    composite[..., 1] = z_img   # green

    composite = np.clip(composite, 0, 1)

    # -----------------------------------------
    # Figure layout
    # -----------------------------------------

    fig, axes = plt.subplots(1, 7, figsize=(28, 4))

    # -----------------------------------------
    # Panel 1 — Original image
    # -----------------------------------------

    if img.ndim == 3 and img.shape[0] == 4:
        display_img = img[0]  
    else:
        display_img = img
    
    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # -----------------------------------------
    # Panel 2 — Composite
    # -----------------------------------------

    axes[1].imshow(composite)
    axes[1].set_title("Composite (B red / Z green)")
    axes[1].axis("off")

    # -----------------------------------------
    # Panel 3 — Nuclei + exclusion
    # -----------------------------------------

    overlay = np.zeros((*nuclei_mask.shape, 3))

    overlay[..., 0] = exclusion_zone
    overlay[..., 2] = nuclei_mask

    axes[2].imshow(overlay)
    axes[2].set_title("Nuclei + Exclusion")
    axes[2].axis("off")

    # -----------------------------------------
    # Panel 4 — Raw B-DNA
    # -----------------------------------------

    axes[3].imshow(b_img, cmap="Reds")
    axes[3].set_title("Raw B-DNA")
    axes[3].axis("off")

    # -----------------------------------------
    # Panel 5 — Processed B-DNA
    # -----------------------------------------

    axes[4].imshow(b_mask, cmap="Reds")
    axes[4].set_title("Processed B-DNA")
    axes[4].axis("off")

    # -----------------------------------------
    # Panel 6 — Raw Z-DNA
    # -----------------------------------------

    axes[5].imshow(z_img, cmap="Greens")
    axes[5].set_title("Raw Z-DNA")
    axes[5].axis("off")

    # -----------------------------------------
    # Panel 7 — Processed Z-DNA
    # -----------------------------------------

    axes[6].imshow(z_mask, cmap="Greens")
    axes[6].set_title("Processed Z-DNA")
    axes[6].axis("off")

    filename = results.get("filename", Path(image_path).name)
    treatment = results.get("treatment", "")

    plt.suptitle(
        f"{filename} {treatment}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    out_dir = Path(image_path).parent / "analysis_output"
    out_dir.mkdir(exist_ok=True)

    save_path = out_dir / f"{Path(image_path).stem}_multichannel.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    
def separate_cells_from_nets(total_dna_mask, cell_diameter=15):
    labeled = measure.label(total_dna_mask)
    regions = measure.regionprops(labeled)
    intact_cells_mask = np.zeros_like(total_dna_mask, dtype=bool)
    nets_mask = np.zeros_like(total_dna_mask, dtype=bool)
    max_cell_area = np.pi * (cell_diameter * 0.8) ** 2
    
    for region in regions:
        area = region.area
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        eccentricity = region.eccentricity
        solidity = region.solidity
        
        is_compact_cell = (area < max_cell_area and aspect_ratio < 2.0 and 
                          solidity > 0.75 and eccentricity < 0.8)
        
        if is_compact_cell:
            intact_cells_mask[labeled == region.label] = True
        else:
            nets_mask[labeled == region.label] = True
    
    return intact_cells_mask, nets_mask


    
def calculate_filamentousness(mask):
    if not np.any(mask):
        return 0.0
    
    skeleton = morphology.skeletonize(mask)
    skeleton_length = np.sum(skeleton)
    area = np.sum(mask)
    filamentousness = skeleton_length / np.sqrt(area) if area > 0 else 0
    
    return filamentousness


def compute_z_spacing_along_skeleton(z_mask, b_skeleton):

    z_on_skel = z_mask & morphology.dilation(b_skeleton, morphology.disk(1))

    labeled = measure.label(z_on_skel)
    regions = measure.regionprops(labeled)

    if len(regions) < 2:
        return np.nan

    skel_coords = np.column_stack(np.nonzero(b_skeleton))

    tree = cKDTree(skel_coords)

    domain_positions = []

    for r in regions:

        if r.area < 5:
            continue

        centroid = r.centroid

        _, idx = tree.query(centroid)

        domain_positions.append(idx)

    if len(domain_positions) < 2:
        return np.nan

    domain_positions = sorted(domain_positions)

    spacing = np.diff(domain_positions)

    if len(spacing) == 0:
        return np.nan

    return float(np.mean(spacing))


    
def determine_treatment(filename, treatment_groups):
    filename_lower = filename.lower()
    
    pattern_list = []
    
    for treatment, patterns in treatment_groups.items():
        for pattern in patterns:
            pattern_list.append((treatment, pattern.lower()))
    
    pattern_list.sort(key=lambda x: len(x[1]), reverse=True)
    
    for treatment, pattern in pattern_list:
        if pattern in filename_lower:
            return treatment
    
    return 'Unknown'



def create_exclusion_mask(nuclei_mask, expansion_radius=5):
    exclusion_mask = morphology.dilation(nuclei_mask, morphology.disk(expansion_radius))
    exclusion_mask = ndimage.binary_fill_holes(exclusion_mask)
    return exclusion_mask


def extract_timepoint(filename):
    match = re.search(r"T(\d+)", filename)

    if match:
        return int(match.group(1))
    else:
        return np.nan


def identify_intact_nuclei_conservative(dna_mask, cell_diameter=15, is_control=False):
    """
    Conservative nuclear identification - especially important for controls.
    In controls (untreated), ALL DNA should be intracellular.
    
    Parameters:
    -----------
    dna_mask : array
        Binary mask of DNA signal
    cell_diameter : int
        Expected cell diameter
    is_control : bool
        If True, assumes NO NETs present (all DNA is nuclear)
    """
    if not np.any(dna_mask):
        return np.zeros_like(dna_mask, dtype=bool), np.zeros_like(dna_mask, dtype=bool)
    
    labeled = measure.label(dna_mask)
    regions = measure.regionprops(labeled)
    
    nuclei_mask = np.zeros_like(dna_mask, dtype=bool)
    nets_mask = np.zeros_like(dna_mask, dtype=bool)
    
    
    min_nucleus_area = np.pi * (cell_diameter * 0.25) ** 2
    max_nucleus_area = np.pi * (cell_diameter * 1.2) ** 2
    
    if is_control:
        max_nucleus_area = np.pi * (cell_diameter * 2.0) ** 2 
        max_aspect_ratio = 6.0 
        min_solidity = 0.4 
    else:
        max_aspect_ratio = 4.0
        min_solidity = 0.55
    
    for region in regions:
        area = region.area
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        solidity = region.solidity
        eccentricity = region.eccentricity
        
        if is_control:
            is_definite_net = (
                major_axis > cell_diameter * 3.0 and  
                aspect_ratio > 8.0 and  
                area > max_nucleus_area  
            )
            
            if is_definite_net:
                nets_mask[labeled == region.label] = True
            else:
                nuclei_mask[labeled == region.label] = True
        
        else:
            is_compact_nucleus = (
                min_nucleus_area < area < max_nucleus_area and
                aspect_ratio < max_aspect_ratio and
                solidity > min_solidity
            )
            
            is_likely_net = (
                (major_axis > cell_diameter * 2.5 and aspect_ratio > 5.0) or 
                (area > max_nucleus_area * 1.5 and solidity < 0.5) or 
                (major_axis > cell_diameter * 3.0)
            )
            
            if is_likely_net and not is_compact_nucleus:
                nets_mask[labeled == region.label] = True
            else:
                nuclei_mask[labeled == region.label] = True
    
    return nuclei_mask, nets_mask


def analyze_filaments(mask, intensity_img, min_length=20):
    
    if not np.any(mask):
        return {
            'num_filaments': 0,
            'mean_length': 0,
            'median_length': 0,
            'max_length': 0,
            'total_skeleton_length': 0,
            'individual_lengths': [],
            'skeleton': np.zeros_like(mask, dtype=bool)
        }

    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    filtered_mask = np.zeros_like(mask, dtype=bool)
    lengths = []

    for region in regions:

        major = region.major_axis_length
        minor = max(region.minor_axis_length, 1)

        elongation = major / minor

        if major >= min_length and elongation > 3:

            lengths.append(major)

            coords = region.coords
            filtered_mask[coords[:,0], coords[:,1]] = True

    skeleton = morphology.skeletonize(filtered_mask)

    return {
        'num_filaments': len(lengths),
        'mean_length': np.mean(lengths) if lengths else 0,
        'median_length': np.median(lengths) if lengths else 0,
        'max_length': np.max(lengths) if lengths else 0,
        'total_skeleton_length': int(np.sum(skeleton)),
        'individual_lengths': lengths,
        'skeleton': skeleton
    }


def classify_skeleton_pixels(skeleton):

    kernel = np.array([
        [1,1,1],
        [1,10,1],
        [1,1,1]
    ])

    neighbor_count = convolve(skeleton.astype(int), kernel, mode="constant")

    endpoints = (neighbor_count == 11)
    linear = (neighbor_count == 12)
    branchpoints = (neighbor_count >= 13)

    return endpoints, linear, branchpoints

def classify_z_relative_to_b(channel_data, all_filaments):

    z_mask = channel_data["Z-DNA"]["mask"]
    b_mask = channel_data["B-DNA"]["mask"]
    b_skel = all_filaments["B-DNA"]["skeleton"]

    skel_zone = morphology.dilation(b_skel, morphology.disk(1))
    near_zone = morphology.dilation(b_mask, morphology.disk(5))

    z_on_backbone = z_mask & skel_zone
    z_near_backbone = z_mask & near_zone & ~skel_zone
    z_independent = z_mask & ~near_zone

    return z_on_backbone, z_near_backbone, z_independent

def analyze_image_multichannel(image_path, channels_config, treatment_groups,
                               cell_diameter=15, min_filament_length=20,
                               exclusion_radius=5, save_output=True):

    filename = Path(image_path).name
    timepoint = extract_timepoint(filename)
    is_control = is_control_file(filename)

    
    if str(image_path).lower().endswith(".tif"):
        
        img = tiff.imread(image_path)
        
        if img.ndim == 3 and img.shape[0] == 4:
            is_tiff_stack = True
        else:
            raise ValueError(f"Unexpected TIFF shape: {img.shape}")
    
    else:
        img = imread(image_path)
        is_tiff_stack = False
    
    
    if not is_tiff_stack:
        
        img = img.astype(np.float32)
        
        if img.max() > 1:
            img = img / 255.0
        
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        
        img = (img * 255).astype(np.uint8)

    channel_data = {}
    
    if is_tiff_stack:
    
        tiff_map = {
            "B-DNA": img[0],
            "Z-DNA": img[1],
            "ZBP1": img[2],
            "MITO": img[3],
        }
        
        for name, config in channels_config.items():
            
            if name not in tiff_map:
                continue
            
            # -----------------------------
            # RAW IMAGE (same as JPG path)
            # -----------------------------
            raw_img = tiff_map[name].astype(float) / 255
            
            # -----------------------------
            # Background subtraction
            # -----------------------------
            background_removed = raw_img.copy()
            
            # -----------------------------
            # SEGMENTATION IMAGE
            # -----------------------------
            p1, p99 = np.percentile(background_removed, (1, 99))
            
            if p99 - p1 > 0:
                seg_img = (background_removed - p1) / (p99 - p1)
            else:
                seg_img = np.zeros_like(background_removed)
            
            # -----------------------------
            # QUANT IMAGE
            # -----------------------------
            quant_img = background_removed.copy()
            
            # -----------------------------
            # Thresholding
            # -----------------------------
            thresh = 0.15
            mask = seg_img > thresh
            
            # -----------------------------
            # Edge filtering
            # -----------------------------            
            edges = sobel(quant_img / (quant_img.max() + 1e-6))
            
            edge_thresh = np.percentile(edges[edges > 0], 75) if np.any(edges > 0) else 0
            
            mask = mask & (edges > edge_thresh)
            
            # -----------------------------
            # Morphology cleanup
            # -----------------------------
            mask = morphology.opening(mask, morphology.disk(1))
            mask = morphology.remove_small_objects(mask, min_size=75)
            
            # -----------------------------
            # Filament filtering
            # -----------------------------
            labeled = label(mask)
            refined_mask = np.zeros_like(mask)
            
            for region in regionprops(labeled):
                if region.eccentricity > 0.85 and region.area > 40:
                    refined_mask[labeled == region.label] = True
            
            mask = refined_mask
            
            # -----------------------------
            # STORE 
            # -----------------------------
            channel_data[name] = {
                'seg_image': seg_img,
                'quant_image': quant_img,
                'mask': mask,
                'config': config
            }
    else:
        for name, config in channels_config.items():
            ch_idx = config['index']

            if ch_idx >= img.shape[-1]:
                continue
            
            raw_img = img[:, :, ch_idx].astype(float) / 255    
            background_removed = raw_img.astype(float)
            # ----------------------------------------
            # IMAGE SEGMENTATION
            # ----------------------------------------
            p1, p99 = np.percentile(background_removed, (1, 99))
            
            if p99 - p1 > 0:
                seg_img = (background_removed - p1) / (p99 - p1)
            else:
                seg_img = np.zeros_like(background_removed)
                
            # ----------------------------------------
            # QUANTIFICATION IMAGE
            # ----------------------------------------
            quant_img = background_removed.copy()
    
            # ----------------------------------------
            # Intensity threshold 
            # ----------------------------------------
            thresh = 0.15   
            mask = seg_img > thresh
            
            # ----------------------------------------
            # Structure filter
            # ----------------------------------------
            
            edges = sobel(quant_img / (quant_img.max() + 1e-6))
            
            edge_thresh = np.percentile(edges[edges > 0], 75) if np.any(edges > 0) else 0
            
            mask = mask & (edges > edge_thresh)
            
            # ----------------------------------------
            # Morphological cleanup
            # ----------------------------------------
            mask = morphology.opening(mask, morphology.disk(1))
            mask = morphology.remove_small_objects(mask, min_size=75) 
    
            # ----------------------------------------
            # Filament-like filtering
            # ----------------------------------------
            labeled = label(mask)
            refined_mask = np.zeros_like(mask)
    
            for region in regionprops(labeled):
                if region.eccentricity > 0.85 and region.area > 40:
                    refined_mask[labeled == region.label] = True
    
            mask = refined_mask
    
            channel_data[name] = {
                'seg_image': seg_img,
                'quant_image': quant_img,
                'mask': mask,
                'config': config
            }    
        
    # ----------------------------------------
    # Nuclear identification
    # ----------------------------------------
     
    primary_dna = 'B-DNA' if 'B-DNA' in channel_data else list(channel_data.keys())[0]
    
    b_img = channel_data[primary_dna]["quant_image"]
    
    b_norm = b_img.astype(float)
    b_norm /= (b_norm.max() + 1e-6)
    
    th = threshold_otsu(b_norm)
    primary_mask = b_norm > th
    
    primary_mask = morphology.opening(primary_mask, morphology.disk(2))
    primary_mask = morphology.remove_small_objects(primary_mask, min_size=100)
    
    all_nuclei, _ = identify_intact_nuclei_conservative(
        primary_mask, cell_diameter, is_control
    )
    exclusion_zone = create_exclusion_mask(
        all_nuclei, expansion_radius=exclusion_radius
    )

    distance_map = distance_transform_edt(~all_nuclei)
    
    # ----------------------------------------
    # NET region definition
    # ----------------------------------------
    total_dna = np.zeros_like(primary_mask, dtype=bool)

    for name, data in channel_data.items():
        total_dna |= data['mask']

    if is_control:
        exclusion_zone = morphology.dilation(
            total_dna, morphology.disk(exclusion_radius)
        )
        net_regions = np.zeros_like(total_dna, dtype=bool)
    else:
        net_regions = total_dna & ~exclusion_zone
        net_regions = morphology.remove_small_objects(net_regions, min_size=100)

    
    # ----------------------------------------
    # Initialize results
    # ----------------------------------------
    results = {
        'filename': filename,
        'is_control': is_control,
        'timepoint': timepoint,
        'num_intact_nuclei': int(measure.label(all_nuclei).max()),
        'total_dna_area_px': int(np.sum(total_dna)),
        'excluded_area_px': int(np.sum(exclusion_zone)),
        'net_area_px': int(np.sum(net_regions)),
    }

    all_filaments = {}

    # ==========================================
    # INTENSITY-BASED Z/B FRACTION OF NET DNA
    # ==========================================
    
    if "Z-DNA" in channel_data and "B-DNA" in channel_data:

        b = channel_data["B-DNA"]["quant_image"]
    
        z = channel_data["Z-DNA"]["quant_image"]
        
        z_img = channel_data["Z-DNA"]["quant_image"]
        b_img = channel_data["B-DNA"]["quant_image"]
    
        z_mask = channel_data["Z-DNA"]["mask"]
        b_mask = channel_data["B-DNA"]["mask"]
    
        dna_backbone = z_mask | b_mask
        dna_net = dna_backbone & net_regions
    
        z_intensity = np.sum(z_img[dna_net])
        b_intensity = np.sum(b_img[dna_net])
    
        total = z_intensity + b_intensity + 1e-6
    
        results["Z_backbone_fraction_intensity"] = z_intensity / total
        results["B_backbone_fraction_intensity"] = b_intensity / total
        results["Total_NET_DNA_intensity"] = total
            
    # ==========================================
    # PER-CHANNEL ANALYSIS
    # ==========================================
    for name, data in channel_data.items():

        channel_nets = data["mask"] & net_regions
        channel_nets = morphology.remove_small_objects(channel_nets, min_size=30)
        fil_data = analyze_filaments(channel_nets, data["quant_image"])

        all_filaments[name] = fil_data

        skeleton = fil_data.get("skeleton", None)

        results[f'{name}_num_filaments'] = fil_data['num_filaments']
        results[f'{name}_mean_length_px'] = float(fil_data['mean_length'])
        results[f'{name}_median_length_px'] = float(fil_data['median_length'])
        results[f'{name}_max_length_px'] = float(fil_data['max_length'])
        results[f'{name}_total_skeleton_length_px'] = fil_data['total_skeleton_length']

        if fil_data['total_skeleton_length'] > 0:
            results[f'{name}_fragmentation_index'] = (
                fil_data['num_filaments'] /
                fil_data['total_skeleton_length']
            )
        else:
            results[f'{name}_fragmentation_index'] = 0.0


        # ----------------------------------------
        # Per-filament Z enrichment
        # ----------------------------------------
        if name == "Z-DNA" and "B-DNA" in channel_data:
            z_mask = channel_nets
            labeled = measure.label(z_mask)
            regions = measure.regionprops(labeled)
            b_img = channel_data["B-DNA"]["quant_image"]
            z_img = data["quant_image"]
            filament_ratios = []
            for region in regions:
                coords = region.coords
                if len(coords) >= min_filament_length:
                    z_vals = [z_img[c[0], c[1]] for c in coords]
                    b_vals = [b_img[c[0], c[1]] for c in coords]
                    z_mean = np.mean(z_vals)
                    b_mean = np.mean(b_vals)
                    ratio = z_mean / (b_mean + 1e-6)
                    filament_ratios.append(ratio)

            if filament_ratios:
                results["Z_filament_enrichment_mean"] = float(np.mean(filament_ratios))
                results["Z_filament_enrichment_median"] = float(np.median(filament_ratios))
            else:
                results["Z_filament_enrichment_mean"] = 0.0
                results["Z_filament_enrichment_median"] = 0.0
        
        
        # ----------------------------------------
        # INTENSITY-BASED METRICS
        # ----------------------------------------
        net_pixels = channel_nets
        if np.sum(net_pixels) > 0:
            results[f'{name}_net_integrated_intensity'] = float(
                np.sum(data['quant_image'][net_pixels])
            )
            results[f'{name}_net_mean_intensity'] = float(
                np.mean(data['quant_image'][net_pixels])
            )
        else:
            results[f'{name}_net_integrated_intensity'] = 0.0
            results[f'{name}_net_mean_intensity'] = 0.0

        results[f'{name}_net_area_px'] = int(np.sum(channel_nets))

        results[f'{name}_filamentousness'] = float(
            calculate_filamentousness(channel_nets)
        )
    # ==========================================
    # NET-RESTRICTED PAIRWISE COLOCALIZATION
    # ==========================================
    channel_names = list(channel_data.keys())

    if len(channel_names) >= 2:
        for i, name1 in enumerate(channel_names):
            for name2 in channel_names[i+1:]:

                mask1_net = channel_data[name1]['mask'] & net_regions
                mask2_net = channel_data[name2]['mask'] & net_regions

                overlap = mask1_net & mask2_net
                union = mask1_net | mask2_net

                overlap_area = np.sum(overlap)
                union_area = np.sum(union)

                area1 = np.sum(mask1_net)
                area2 = np.sum(mask2_net)

                jaccard = overlap_area / union_area if union_area > 0 else 0
                manders_m1 = overlap_area / area1 if area1 > 0 else 0
                manders_m2 = overlap_area / area2 if area2 > 0 else 0

                prefix = f'{name1}_vs_{name2}'

                results[f'{prefix}_jaccard'] = float(jaccard)
                results[f'{prefix}_manders_{name1}'] = float(manders_m1)
                results[f'{prefix}_manders_{name2}'] = float(manders_m2)
                results[f'{prefix}_overlap_area_px'] = int(overlap_area)

    
    # ==========================================
    # SKELETON CONTRIBUTION BETWEEN CHANNELS
    # ==========================================
    if len(channel_names) >= 2:
        for i, name1 in enumerate(channel_names):
            for name2 in channel_names[i+1:]:

                total_skel = (
                    all_filaments[name1]['total_skeleton_length'] +
                    all_filaments[name2]['total_skeleton_length']
                )

                if total_skel > 0:
                    results[f'{name1}_vs_{name2}_contrib_{name1}'] = float(
                        all_filaments[name1]['total_skeleton_length'] / total_skel
                    )
                    results[f'{name1}_vs_{name2}_contrib_{name2}'] = float(
                        all_filaments[name2]['total_skeleton_length'] / total_skel
                    )


    # ==========================================
    # Z DOMAIN NUMBER and LENGTH
    # ==========================================
    
    if "Z-DNA" in channel_data and "B-DNA" in channel_data:
    
        z_mask = channel_data["Z-DNA"]["mask"]
        b_skel = all_filaments["B-DNA"]["skeleton"]
    
        z_on_skel = z_mask & b_skel
    
        labeled_domains = label(z_on_skel)
        regions = regionprops(labeled_domains)
    
        endpoints, linear, branchpoints = classify_skeleton_pixels(
            all_filaments["B-DNA"]["skeleton"]
        )
    
        domain_lengths = []
    
        for r in regions:
    
            coords = tuple(zip(*r.coords))
    
            if len(coords[0]) <= 2:
                continue
    
            domain_lengths.append(len(coords[0]))
            
        if len(domain_lengths) > 0:
    
            results["Z_mean_domain_length"] = float(np.mean(domain_lengths))
            results["Z_max_domain_length"] = float(np.max(domain_lengths))
            results["Z_num_domains"] = int(len(domain_lengths))
    
        else:
    
            results["Z_mean_domain_length"] = 0
            results["Z_max_domain_length"] = 0
            results["Z_num_domains"] = 0
            
    # ==========================================
    # STRUCTURAL DOMINANCE SUPPORT METRICS
    # ==========================================

    if "Z-DNA" in channel_data and "B-DNA" in channel_data:

        z_skel = all_filaments["Z-DNA"]["skeleton"]
        b_skel = all_filaments["B-DNA"]["skeleton"]

        z_total = np.sum(z_skel)
        b_total = np.sum(b_skel)

        overlap = z_skel & b_skel
        occupancy = np.sum(overlap) / (b_total + 1e-6)

        z_num = all_filaments["Z-DNA"]["num_filaments"]
        z_mean_continuity = z_total / (z_num + 1e-6)

        skeleton_ratio = z_total / (z_total + b_total + 1e-6)

        fragmentation_penalty = z_num / (z_total + 1e-6)

        results["Z_occupancy"] = float(occupancy)
        results["Z_mean_continuity"] = float(z_mean_continuity)
        results["Z_skeleton_ratio"] = float(skeleton_ratio)
        results["Z_fragmentation_penalty"] = float(fragmentation_penalty)
        if b_total == 0:
            results["log_ZB"] = 0
        else:
            results["log_ZB"] = np.log10((z_total + 1e-6) / (b_total + 1e-6))
        results["Z_efficiency"] = z_total / (z_num + 1e-6)

    z_on, z_near, z_indep = classify_z_relative_to_b(channel_data, all_filaments)

    total_z = np.sum(channel_data["Z-DNA"]["mask"]) + 1
    
    results["Z_on_B_fraction"] = np.sum(z_on) / total_z
    results["Z_near_B_fraction"] = np.sum(z_near) / total_z
    results["Z_independent_fraction"] = np.sum(z_indep) / total_z

    # ==========================================
    # 8oxoG ANALYSIS (DNA DAMAGE)
    # ==========================================
    
    if "8oxoG" in channel_data and "Z-DNA" in channel_data:
    
        oxo_mask = channel_data["8oxoG"]["mask"]
        oxo_img  = channel_data["8oxoG"]["quant_image"]
    
        z_mask = channel_data["Z-DNA"]["mask"]
        b_skel = all_filaments["B-DNA"]["skeleton"]
    
        oxo_net = oxo_mask & net_regions
    
        results["8oxoG_total_intensity"] = float(np.sum(oxo_img[oxo_net]))
        results["8oxoG_area"] = int(np.sum(oxo_net))
    
        results["8oxoG_density"] = (
            results["8oxoG_total_intensity"] /
            (np.sum(net_regions) + 1e-6)
        )
    
        overlap = oxo_net & z_mask
    
        results["8oxoG_on_Z_fraction"] = (
            np.sum(overlap) / (np.sum(oxo_net) + 1e-6)
        )
        z_on_oxo = z_mask & oxo_mask
    
        results["Z_on_8oxo_fraction"] = (
            np.sum(z_on_oxo) / (np.sum(z_mask) + 1e-6)
        )
        results["8oxoG_Z_colocalization"] = (
            results["8oxoG_on_Z_fraction"] +
            results["Z_on_8oxo_fraction"]
        ) / 2    

        dist_to_z = distance_transform_edt(~z_mask)
        
        dist_to_oxo = distance_transform_edt(~oxo_mask)
        
        oxo_to_z = dist_to_z[oxo_mask]
        z_to_oxo = dist_to_oxo[z_mask]
        
        results["8oxoG_to_Z_distance"] = float(np.mean(oxo_to_z)) if len(oxo_to_z) > 0 else 0
        results["Z_to_8oxo_distance"] = float(np.mean(z_to_oxo)) if len(z_to_oxo) > 0 else 0
        results["8oxoG_Z_proximity_score"] = np.exp(
            -results["8oxoG_to_Z_distance"] / 200 
        )        
        z_img = channel_data["Z-DNA"]["quant_image"]
        oxo_img = channel_data["8oxoG"]["quant_image"]
        
        z_vals = z_img[net_regions].flatten()
        oxo_vals = oxo_img[net_regions].flatten()

        if len(z_vals) > 10:
            results["8oxoG_Z_intensity_corr"] = np.corrcoef(z_vals, oxo_vals)[0,1]
        else:
            results["8oxoG_Z_intensity_corr"] = 0

            
        from skimage.morphology import dilation, disk
    
        z_dilated = dilation(z_mask, disk(2))
    
        near_z = oxo_net & z_dilated
    
        results["8oxoG_near_Z_fraction"] = (
            np.sum(near_z) / (np.sum(oxo_net) + 1e-6)
        )
        
        dist_map = distance_transform_edt(~b_skel)
    
        oxo_dist = dist_map[oxo_net]
    
        if len(oxo_dist) > 5:
            results["8oxoG_mean_distance_to_B"] = float(np.mean(oxo_dist))
        else:
            results["8oxoG_mean_distance_to_B"] = 0
    
    else:
        results["8oxoG_total_intensity"] = 0
        results["8oxoG_area"] = 0
        results["8oxoG_density"] = 0
        results["8oxoG_on_Z_fraction"] = 0
        results["8oxoG_near_Z_fraction"] = 0
        results["8oxoG_mean_distance_to_B"] = 0

    # ==========================================
    # Z-DNA mitochondrial vs non-mitochondrial
    # ==========================================
    
    results["Z_mito_fraction"] = np.nan
    results["Z_non_mito_fraction"] = np.nan
    
    if "MITO" in channel_data:
        mito_mask = channel_data["MITO"]["mask"]
        
        if "Z-DNA" in channel_data:
            z_mask = channel_data["Z-DNA"]["mask"]
            net_mask = net_regions > 0
        
            z_net = z_mask & net_mask
        
            if np.sum(z_net) > 0:
        
                z_mito = np.sum(z_net & mito_mask)
                z_non_mito = np.sum(z_net & (~mito_mask))
        
                total = z_mito + z_non_mito
        
                if total > 0:
                    results["Z_mito_fraction"] = z_mito / total
                    results["Z_non_mito_fraction"] = z_non_mito / total
        
        if "B-DNA" in channel_data:
            b_mask = channel_data["B-DNA"]["mask"]
            results["B_mito_fraction"] = np.sum(b_mask & mito_mask) / (np.sum(b_mask) + 1e-6)

    # ==========================================
    # ZBP1 colocalization with mito vs non-mito Z-DNA
    # ==========================================
    
    results["ZBP1_Manders_mito_M1"] = np.nan
    results["ZBP1_Manders_mito_M2"] = np.nan
    results["ZBP1_Jaccard_mito"] = np.nan
    
    results["ZBP1_Manders_non_mito_M1"] = np.nan
    results["ZBP1_Manders_non_mito_M2"] = np.nan
    results["ZBP1_Jaccard_non_mito"] = np.nan
    
    if "ZBP1" in channel_data and "Z-DNA" in channel_data and "MITO" in channel_data:
    
        zbp = channel_data["ZBP1"]["mask"]
        z = channel_data["Z-DNA"]["mask"]
        mito = channel_data["MITO"]["mask"]
    
        net_mask = net_regions > 0
    
        z_net = z & net_mask
        zbp_net = zbp & net_mask
    
        z_mito = z_net & mito
        z_non_mito = z_net & (~mito)
    
        # ------------------------------------------
        # MITO Z-DNA
        # ------------------------------------------
        inter_mito = np.sum(zbp_net & z_mito)
        union_mito = np.sum(zbp_net | z_mito)
    
        if np.sum(zbp_net) > 0:
            results["ZBP1_Manders_mito_M1"] = inter_mito / np.sum(zbp_net)
    
        if np.sum(z_mito) > 0:
            results["ZBP1_Manders_mito_M2"] = inter_mito / np.sum(z_mito)
    
        if union_mito > 0:
            results["ZBP1_Jaccard_mito"] = inter_mito / union_mito
    
        # ------------------------------------------
        # NON-MITO Z-DNA
        # ------------------------------------------
        inter_non = np.sum(zbp_net & z_non_mito)
        union_non = np.sum(zbp_net | z_non_mito)
    
        if np.sum(zbp_net) > 0:
            results["ZBP1_Manders_non_mito_M1"] = inter_non / np.sum(zbp_net)
    
        if np.sum(z_non_mito) > 0:
            results["ZBP1_Manders_non_mito_M2"] = inter_non / np.sum(z_non_mito)
    
        if union_non > 0:
            results["ZBP1_Jaccard_non_mito"] = inter_non / union_non

    # ----------------------------------------
    # Save visualization
    # ----------------------------------------
    if save_output:
        save_multichannel_visualization(
            img,
            channel_data,
            all_nuclei,
            exclusion_zone,
            net_regions,
            all_filaments,
            results,
            image_path
        )

    return results, all_filaments, channel_data


