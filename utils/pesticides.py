"""Simple treatment lookup for the checkpoint's exact dataset labels.

Research sources and limitations are recorded in the project README.
"""

PESTICIDES = {
    'Bacterial Leaf Blight': 'Copper hydroxide',
    'Brown Spot': 'Mancozeb',
    'Healthy Rice Leaf': 'None required',
    'Leaf Blast': 'Tricyclazole',
    'Leaf scald': 'Validamycin',
    'Narrow Brown Leaf Spot': 'Propiconazole',
    'Rice Hispa': 'Malathion',
    # Preserve the spelling used by the trained model for rice tungro.
    'Rice Turgro': 'No pesticide cures tungro; manage leafhopper vectors',
    'Sheath Blight': 'Validamycin',
}
