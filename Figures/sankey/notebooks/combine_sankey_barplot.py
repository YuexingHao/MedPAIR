#!/usr/bin/env python3
"""Combine Sankey and Barplot figures vertically into one image."""

from pathlib import Path
from PIL import Image

def main():
    repo = Path(__file__).resolve().parents[3]
    fig_dir = repo / "Figures" / "sankey" / "figures"
    
    # Load both images
    sankey_path = fig_dir / "llm_sankey_combined.png"
    barplot_path = fig_dir / "llm_barplot_sankey_combined.png"
    
    print(f"Loading Sankey from {sankey_path}")
    sankey_img = Image.open(sankey_path)
    
    print(f"Loading Barplot from {barplot_path}")
    barplot_img = Image.open(barplot_path)
    
    # Get dimensions
    sankey_w, sankey_h = sankey_img.size
    barplot_w, barplot_h = barplot_img.size
    
    print(f"Sankey size: {sankey_w}x{sankey_h}")
    print(f"Barplot size: {barplot_w}x{barplot_h}")
    
    # Use the maximum width and add heights
    combined_w = max(sankey_w, barplot_w)
    combined_h = sankey_h + barplot_h
    
    # Create new image with white background
    combined_img = Image.new('RGB', (combined_w, combined_h), 'white')
    
    # Paste images
    # Center both images if widths differ
    sankey_x = (combined_w - sankey_w) // 2
    barplot_x = (combined_w - barplot_w) // 2
    
    combined_img.paste(sankey_img, (sankey_x, 0))
    combined_img.paste(barplot_img, (barplot_x, sankey_h))
    
    # Save combined image
    output_path = fig_dir / "llm_sankey_barplot_combined.png"
    combined_img.save(output_path, dpi=(220, 220))
    print(f"Wrote combined figure to {output_path}")
    print(f"Combined size: {combined_w}x{combined_h}")

if __name__ == "__main__":
    main()
