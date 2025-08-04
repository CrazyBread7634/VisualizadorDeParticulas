import { 
    initViewer, 
    loadMolecule, 
    update3DStyle, 
    toggle3DLabels, 
    getViewer,
    rerenderCurrentMolecule 
} from './viewer.js';
import { 
    populateMoleculeCards, 
    updateCardSelection, 
    updateButtonState,
    initContextMenu 
} from './ui.js';
import { setupDragAndDrop, clearCombination, combineMolecules } from './combination.js';
import { saveCompound, loadCompounds, deleteCompound } from './firebase/db.js';

let currentSelectedMolecule = 'estradiol';
let compoundToDelete = null;

function handleDownload() {
    const viewer = getViewer();
    if (!viewer || !viewer.getModel()) {
        return;
    }
    const format = document.getElementById('download-format-select').value;
    const model = viewer.getModel();
    const scene = new THREE.Scene();
}

function downloadBlob(data, filename, mimeType) {
    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function handleSaveCompound() {
    const resultSection = document.getElementById('combination-result-section');
    const compoundData = {
        name: resultSection.dataset.name,
        smiles: resultSection.dataset.smiles,
        analysis: resultSection.querySelector('#ai-analysis-content').innerHTML,
        createdAt: new Date()
    };

    const button = document.getElementById('save-compound-btn');
    button.disabled = true;
    button.textContent = 'Guardando...';

    try {
        await saveCompound(compoundData);
        button.textContent = 'Guardado';
        renderSavedCompounds();
    } catch (error) {
        alert('Error al guardar el compuesto.');
        button.disabled = false;
        button.textContent = 'Guardar Compuesto';
    }
}

function showDeleteConfirmation(compound) {
    compoundToDelete = compound;
    const popup = document.getElementById('delete-confirmation-popup');
    const compoundNameElement = popup.querySelector('.compound-name-to-delete');
    compoundNameElement.textContent = `"${compound.name}"`;
    popup.style.display = 'flex';
}

function hideDeleteConfirmation() {
    const popup = document.getElementById('delete-confirmation-popup');
    popup.style.display = 'none';
    compoundToDelete = null;
}

async function handleDeleteCompound() {
    if (!compoundToDelete) return;
    
    const confirmButton = document.getElementById('confirm-delete-btn');
    const originalText = '<i class="fi fi-br-trash"></i> Eliminar';
    
    try {
        confirmButton.disabled = true;
        confirmButton.innerHTML = '<i class="fi fi-br-loading"></i> Eliminando...';
        
        await deleteCompound(compoundToDelete.id);
        
        confirmButton.disabled = false;
        confirmButton.innerHTML = originalText;
        
        hideDeleteConfirmation();
        renderSavedCompounds();
        
    } catch (error) {
        alert('Error al eliminar el compuesto. Por favor, intenta de nuevo.');
        confirmButton.disabled = false;
        confirmButton.innerHTML = originalText;
    }
}

async function renderSavedCompounds() {
    const container = document.getElementById('saved-compounds-container');
    container.innerHTML = 'Cargando...';
    const compounds = await loadCompounds();
    container.innerHTML = '';
    if (compounds.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fi fi-br-empty-set"></i>
                <p>No hay compuestos guardados.</p>
                <span class="empty-hint">Combina moléculas y guarda los resultados.</span>
            </div>
        `;
        return;
    }
    compounds.forEach(compound => {
        const card = document.createElement('div');
        card.className = 'molecule-card saved-compound-card';
        card.innerHTML = `
            <button class="delete-compound-btn" title="Eliminar compuesto">
                <i class="fi fi-br-trash"></i>
            </button>
            <p>${compound.name}</p>
            <div class="card-buttons">
                <button class="control-btn load-btn">Cargar</button>
            </div>
        `;
        
        card.querySelector('.load-btn').addEventListener('click', () => {
            loadMolecule(compound.name, compound.smiles);
            const resultSection = document.getElementById('combination-result-section');
            resultSection.innerHTML = `
                <h3>Análisis del Compuesto Guardado: ${compound.name}</h3>
                <div id="ai-analysis-content">${compound.analysis}</div>
            `;
            resultSection.style.display = 'block';
            resultSection.classList.add('visible');
        });
        
        card.querySelector('.delete-compound-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            showDeleteConfirmation(compound);
        });
        
        container.appendChild(card);
    });
}

function init() {
    initViewer();
    initContextMenu();
    populateMoleculeCards();
    setupDragAndDrop();

    document.querySelectorAll('.molecule-card').forEach(card => {
        card.addEventListener('click', () => {
            const moleculeName = card.dataset.moleculeName;
            currentSelectedMolecule = moleculeName;
            updateCardSelection(moleculeName);
            loadMolecule(moleculeName);
        });
        card.addEventListener('dragstart', (event) => {
            event.dataTransfer.setData('text/plain', card.dataset.moleculeName);
            event.dataTransfer.effectAllowed = 'copy';
            card.classList.add('dragging');
        });
        card.addEventListener('dragend', () => {
            card.classList.remove('dragging');
        });
    });

    document.querySelectorAll('input[name="style3d"], #show-hydrogens').forEach(el => {
        el.addEventListener('change', update3DStyle);
    });

    document.getElementById('show-atom-indices').addEventListener('change', rerenderCurrentMolecule);
    document.getElementById('show-bond-indices').addEventListener('change', rerenderCurrentMolecule);
    document.getElementById('show-atom-indices-3d').addEventListener('change', toggle3DLabels);
    document.getElementById('show-bond-indices-3d').addEventListener('change', toggle3DLabels);
    
    document.getElementById('reset-view-btn').addEventListener('click', () => getViewer()?.zoomTo());
    document.getElementById('context-menu-close').addEventListener('click', () => document.getElementById('context-menu').style.display = 'none');
    document.getElementById('download-btn').addEventListener('click', handleDownload);
    
    const combinationBtn = document.getElementById('clear-combination-btn');
    combinationBtn.addEventListener('click', () => {
        if (combinationBtn.textContent.includes('Combinar')) {
            combineMolecules();
        } else {
            clearCombination(currentSelectedMolecule);
        }
    });

    document.getElementById('combination-result-section').addEventListener('click', (event) => {
        if (event.target && event.target.id === 'save-compound-btn') {
            handleSaveCompound();
        }
    });

    document.getElementById('cancel-delete-btn').addEventListener('click', hideDeleteConfirmation);
    document.getElementById('confirm-delete-btn').addEventListener('click', handleDeleteCompound);
    
    document.getElementById('delete-confirmation-popup').addEventListener('click', (e) => {
        if (e.target.id === 'delete-confirmation-popup') {
            hideDeleteConfirmation();
        }
    });

    loadMolecule(currentSelectedMolecule);
    updateCardSelection(currentSelectedMolecule);
    updateButtonState('Limpiar Combinación', false, true);
    renderSavedCompounds();
}

document.addEventListener('DOMContentLoaded', init);
