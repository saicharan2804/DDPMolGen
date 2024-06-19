import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import argparse

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate molecule images from SMILES strings.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing SMILES strings.")
    parser.add_argument('--smiles_column', type=str, required=True, help="Column name corresponding to SMILES data in the CSV file.")
    parser.add_argument('--output_dir', type=str, default='molecule_images', help="Directory to save the output images.")
    parser.add_argument('--num_samples', type=int, default=8, help="Number of random samples to generate images for.")
    return parser.parse_args()

def generate_molecule_image(smiles: str, image_path: str) -> None:
    """
    Generate an image of a molecule from a SMILES string and save it.

    Args:
        smiles (str): SMILES string representing the molecule.
        image_path (str): Path to save the generated image.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    img = Draw.MolToImage(molecule, size=(300, 300))
    img.save(image_path)
    print(f"Image saved to {image_path}")

def main():
    """
    Main function to execute the script.
    """
    args = parse_arguments()
    
    # Load the DataFrame
    df = pd.read_csv(args.csv_file)

    # Ensure the SMILES column exists
    if args.smiles_column not in df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found in the CSV file.")

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Sample random SMILES strings
    sampled_smiles = df[args.smiles_column].sample(n=args.num_samples, random_state=42)

    # Generate and save images for the sampled molecules
    for i, smiles in enumerate(sampled_smiles):
        image_path = os.path.join(args.output_dir, f'molecule_{i+1}.png')
        generate_molecule_image(smiles, image_path)

    print('All images saved')

if __name__ == "__main__":
    main()