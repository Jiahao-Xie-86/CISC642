import os
import subprocess
import argparse
import time


def run_stereo_analysis(config_id, image_dir, output_dir, stereo_pair, levels, match_method,
                       template_height, template_width, max_disparity, feature_based_levels,
                       patch_size, search_range):
    """
    Run stereo analysis with specified parameters.
    
    Args:
        config_id: Configuration ID for output naming
        image_dir: Directory containing stereo datasets
        output_dir: Directory to save results
        stereo_pair: Stereo pair to process
        levels: Number of pyramid levels
        match_method: Matching method
        template_height: Template height for region-based matching
        template_width: Template width for region-based matching
        max_disparity: Maximum disparity
        feature_based_levels: Comma-separated list of levels to use feature-based matching
        patch_size: Patch size for feature-based matching
        search_range: Search range around initial disparity
    """
    # Create output directory for this configuration
    config_output_dir = os.path.join(output_dir, f"config_{config_id}")
    os.makedirs(config_output_dir, exist_ok=True)
    
    cmd = [
        "python", "stereo_matching.py",
        "--image_dir", image_dir,
        "--output_dir", config_output_dir,
        "--stereo_pair", stereo_pair,
        "--levels", str(levels),
        "--match_method", match_method,
        "--template_height", str(template_height),
        "--template_width", str(template_width),
        "--max_disparity", str(max_disparity),
        "--feature_based_levels", feature_based_levels,
        "--patch_size", str(patch_size),
        "--search_range", str(search_range)
    ]
    
    print(f"Running configuration {config_id}:")
    print(" ".join(cmd))
    
    start_time = time.time()
    subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    print(f"Configuration {config_id} completed in {elapsed_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Run Stereo Analysis System with different configurations')
    
    parser.add_argument('--image_dir', type=str, default='./images',
                        help='Directory containing stereo datasets')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Define configurations to run
    configs = [
        # Configuration 1: Basic region-based matching with NCC
        {
            "id": 1,
            "stereo_pair": "all",
            "levels": 3,
            "match_method": "NCC",
            "template_height": 5,
            "template_width": 5,
            "max_disparity": 64,
            "feature_based_levels": "",
            "patch_size": 11,
            "search_range": 5
        },
        
        # Configuration 2: Multi-resolution with feature-based on middle level
        {
            "id": 2,
            "stereo_pair": "all",
            "levels":4,
            "match_method": "NCC",
            "template_height": 5,
            "template_width": 5,
            "max_disparity": 64,
            "feature_based_levels": "1",  # Use feature-based on middle level
            "patch_size": 11,
            "search_range": 5
        },
        
        # Configuration 3: Region-based with SSD
        {
            "id": 3,
            "stereo_pair": "all",
            "levels": 5,
            "match_method": "SSD",
            "template_height": 5,
            "template_width":5,
            "max_disparity": 64,
            "feature_based_levels": "",
            "patch_size": 11,
            "search_range": 3
        },
        
        # Configuration 4: Higher resolution with smaller template
        {
            "id": 4,
            "stereo_pair": "all",
            "levels": 4,
            "match_method": "NCC",
            "template_height": 5,
            "template_width": 5,
            "max_disparity": 64,
            "feature_based_levels": "",
            "patch_size": 11,
            "search_range": 3
        },
        
        # Configuration 5: SAD with rectangular template
        {
            "id": 5,
            "stereo_pair": "all",
            "levels": 5,
            "match_method": "SAD",
            "template_height": 3,
            "template_width": 5,
            "max_disparity": 64,
            "feature_based_levels": "",
            "patch_size": 11,
            "search_range": 3
        }
    ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run all configurations
    for config in configs:
        run_stereo_analysis(
            config["id"],
            args.image_dir,
            args.output_dir,
            config["stereo_pair"],
            config["levels"],
            config["match_method"],
            config["template_height"],
            config["template_width"],
            config["max_disparity"],
            config["feature_based_levels"],
            config["patch_size"],
            config["search_range"]
        )
    
    print("All configurations completed!")


if __name__ == "__main__":
    main()