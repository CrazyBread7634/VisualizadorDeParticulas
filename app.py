from flask import Flask, render_template, jsonify, request
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
import os
import json

app = Flask(__name__)

# --- Base de datos de moléculas iniciales ---
molecules_db = {
    "estradiol": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    "fulvestrant": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@H](C(F)(F)C(F)(F)S(=O)CCCCCCCCC)C2"
}

def smiles_to_mol_block(smiles):
    """Convierte una cadena SMILES a un bloque MOL 3D optimizado."""
    try:
        # Validación mínima - solo verificar que RDKit pueda parsear
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"RDKit no puede parsear SMILES: {smiles}")
            return None
            
        num_atoms = mol.GetNumAtoms()
        print(f"📊 Procesando molécula con {num_atoms} átomos (longitud SMILES: {len(smiles)})")
            
        mol = Chem.AddHs(mol)  # Añadir hidrógenos
        
        embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if embed_result == -1:
            embed_result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if embed_result == -1:
                print("No se pudo generar conformación 3D")
                return None
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            print("Advertencia: No se pudo optimizar con MMFF94, usando geometría sin optimizar")
            
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block
    except Exception as e:
        print(f"Error al procesar SMILES '{smiles}': {e}")
        return None

def smiles_to_svg(smiles, show_atom_indices=True, show_bond_indices=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        rdDepictor.Compute2DCoords(mol)

        bonds = []
        for bond in mol.GetBonds():
            bonds.append({
                'bond_index': bond.GetIdx(),
                'atom1_index': bond.GetBeginAtomIdx(),
                'atom2_index': bond.GetEndAtomIdx()
            })

        d2d = Draw.MolDraw2DSVG(400, 350)
        opts = d2d.drawOptions()
        opts.addAtomIndices = show_atom_indices
        opts.addBondIndices = show_bond_indices
        opts.clearBackground = False
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        
        return d2d.GetDrawingText(), bonds
    except Exception as e:
        print(f"Error al generar SVG: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/molecule/<molecule_name>')
def get_molecule(molecule_name):
    smiles = molecules_db.get(molecule_name)
    if not smiles:
        return jsonify({"error": "Molécula no encontrada"}), 404

    show_atoms = request.args.get('show_atoms', 'true').lower() == 'true'
    show_bonds = request.args.get('show_bonds', 'true').lower() == 'true'

    mol_block = smiles_to_mol_block(smiles)
    svg_image, bonds_data = smiles_to_svg(smiles, show_atom_indices=show_atoms, show_bond_indices=show_bonds)

    if mol_block and svg_image:
        return jsonify({"mol": mol_block, "smiles": smiles, "svg": svg_image, "bonds": bonds_data})
    else:
        return jsonify({"error": "No se pudo generar la estructura 3D o 2D"}), 500

@app.route('/api/render_smiles', methods=['POST'])
def render_smiles():
    data = request.json
    smiles = data.get('smiles')
    show_atoms = data.get('show_atoms', True)
    show_bonds = data.get('show_bonds', True)

    if not smiles:
        return jsonify({"error": "No se proporcionó SMILES"}), 400
    
    mol_block = smiles_to_mol_block(smiles)
    svg_image, bonds_data = smiles_to_svg(smiles, show_atom_indices=show_atoms, show_bond_indices=show_bonds)

    if mol_block and svg_image:
        return jsonify({"mol": mol_block, "smiles": smiles, "svg": svg_image, "bonds": bonds_data})
    else:
        error_details = []
        if not mol_block:
            error_details.append("No se pudo generar la estructura 3D")
        if not svg_image:
            error_details.append("No se pudo generar la representación 2D")
        
        detailed_error = f"Error procesando SMILES '{smiles}': {', '.join(error_details)}"
        
        return jsonify({
            "error": "El SMILES proporcionado no es válido o no se pudo generar la estructura.",
            "details": detailed_error,
            "smiles": smiles,
            "suggestions": [
                "Verifica que el SMILES no contenga caracteres inválidos",
                "La molécula puede ser demasiado compleja para procesar",
                "Intenta con una estructura más simple",
                "Usa un modelo de IA diferente para generar el SMILES"
            ]
        }), 422

if __name__ == '__main__':
    app.run(debug=True)
