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
import { saveCompound, loadCompounds } from './firebase/db.js';

let currentSelectedMolecule = 'estradiol';

function handleDownload() {
    const viewer = getViewer();
    if (!viewer || !viewer.getModel()) {
        console.error("3D viewer or model not initialized.");
        return;
    }
    const format = document.getElementById('download-format-select').value;
    const model = viewer.getModel();
    const scene = new THREE.Scene();
    // ... (Lógica de exportación 3D, puede ser movida a su propio módulo si crece)
    // Por simplicidad, la mantenemos aquí por ahora.
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
        alert('Compuesto guardado con éxito!');
        button.textContent = 'Guardado';
        renderSavedCompounds();
    } catch (error) {
        alert('Error al guardar el compuesto.');
        button.disabled = false;
        button.textContent = 'Guardar Compuesto';
    }
}

async function renderSavedCompounds() {
    const container = document.getElementById('saved-compounds-container');
    container.innerHTML = 'Cargando...';
    const compounds = await loadCompounds();
    container.innerHTML = '';
    if (compounds.length === 0) {
        container.innerHTML = '<p>No hay compuestos guardados.</p>';
        return;
    }
    compounds.forEach(compound => {
        const card = document.createElement('div');
        card.className = 'molecule-card saved-compound-card';
        card.innerHTML = `
            <p>${compound.name}</p>
            <button class="control-btn load-btn">Cargar</button>
        `;
        card.querySelector('.load-btn').addEventListener('click', () => {
            loadMolecule(compound.name, compound.smiles);
            const resultSection = document.getElementById('combination-result-section');
            resultSection.innerHTML = `
                <h3>Análisis del Compuesto Guardado: ${compound.name}</h3>
                <div id="ai-analysis-content">${compound.analysis}</div>
            `;
             resultSection.style.display = 'block';
        });
        container.appendChild(card);
    });
}

function init() {
    initViewer();
    initContextMenu(); // Inicializar la lógica del menú contextual
    populateMoleculeCards();
    setupDragAndDrop();

    // --- Event Listeners ---
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
        // La función del botón depende de su estado actual
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

    // Carga inicial
    loadMolecule(currentSelectedMolecule);
    updateCardSelection(currentSelectedMolecule);
    updateButtonState('Limpiar Combinación', false, true);
    renderSavedCompounds();
}

document.addEventListener('DOMContentLoaded', init);
