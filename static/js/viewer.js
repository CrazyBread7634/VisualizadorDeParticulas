import { handleAtomClick, handleBondClick } from './ui.js';

let viewer;
let currentSmiles = '';
let currentMolBlock = '';
let currentBonds = [];
let lastHoveredAtom = null;
let panZoomInstance = null;
let currentMoleculeName = ''; // Variable para preservar el nombre

// Elementos a animar
const animatedElements = [
    document.getElementById('molecule-name-2d'),
    document.getElementById('molecule-name-3d'),
    document.getElementById('viewer-2d'),
    document.getElementById('viewer-3d')
];
const viewer2dOverlay = document.querySelector('#viewer-container-2d .viewer-loader-overlay');
const viewer3dOverlay = document.querySelector('#viewer-container-3d .viewer-loader-overlay');

const viewer2DContainer = document.getElementById('viewer-2d');
const moleculeName2d = document.getElementById('molecule-name-2d');
const moleculeName3d = document.getElementById('molecule-name-3d');

export function showViewerLoaders(message = "Combinando Moléculas...") {
    // Aplicar animación de desvanecimiento
    animatedElements.forEach(el => el.classList.add('content-fading'));

    setTimeout(() => {
        // Actualizar títulos
        moleculeName2d.textContent = message;
        moleculeName3d.textContent = message;
        
        const loaderHtml = `
            <div class="loader-container">
                <lord-icon 
                    src="https://cdn.lordicon.com/tewlfgbl.json" 
                    trigger="loop" 
                    state="loop-dispersion"
                    colors="primary:#616161"
                    style="width:100px;height:100px;"></lord-icon>
                <p class="carrusel-text">Combinando...</p>
            </div>
        `;
        
        // Cargar loaders
        viewer2dOverlay.innerHTML = loaderHtml; // Carga el loader en el overlay 2D
        viewer2dOverlay.style.display = 'flex'; // Muestra el overlay 2D
        
        viewer3dOverlay.innerHTML = loaderHtml; // Carga el loader en el overlay 3D
        viewer3dOverlay.style.display = 'flex'; // Muestra el overlay 3D
        
        // Revelar los loaders
        animatedElements.forEach(el => el.classList.remove('content-fading'));
    }, 300); // Sincronizar con la animación
}


export function initViewer() {
    const viewerElement = document.getElementById('viewer-3d');
    viewer = $3Dmol.createViewer(viewerElement, {
        defaultcolors: $3Dmol.elementColors.Jmol,
        backgroundColor: 'white'
    });
    return viewer;
}

export function getViewer() {
    return viewer;
}

export function getCurrentSmiles() {
    return currentSmiles;
}

export function getCurrentBonds() {
    return currentBonds;
}

/**
 * Loads a molecule from the API and updates the 2D and 3D viewers.
 * @param {string} name - The name of the molecule to load.
 */
export async function loadMolecule(name) {
    animatedElements.forEach(el => el.classList.add('content-fading'));

    // Esperar a que la animación de salida comience
    await new Promise(resolve => setTimeout(resolve, 300));

    try {
        const showAtoms = document.getElementById('show-atom-indices').checked;
        const showBonds = document.getElementById('show-bond-indices').checked;
        const response = await fetch(`/api/molecule/${name}?show_atoms=${showAtoms}&show_bonds=${showBonds}`);
        if (!response.ok) {
            throw new Error(`Error del servidor: ${response.status}`);
        }
        const data = await response.json();
        
        loadMoleculeFromData(data, name);
        
        return data;

    } catch (error) {
        console.error('Error en loadMolecule:', error);
        throw error;
    }
}

export async function rerenderCurrentMolecule() {
    if (!currentSmiles) return;

    try {
        const response = await fetch('/api/render_smiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                smiles: currentSmiles, 
                show_atoms: document.getElementById('show-atom-indices').checked, 
                show_bonds: document.getElementById('show-bond-indices').checked 
            })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        const data = await response.json();
        loadMoleculeFromData(data, currentMoleculeName); // Re-renderizar con el nombre guardado
    } catch (error) {
        console.error('Error al re-renderizar la molécula:', error);
    }
}

export function loadMoleculeFromData(data, name = "Nueva Molécula") {
    // Actualizar estado global
    currentSmiles = data.smiles;
    currentMolBlock = data.mol;
    currentBonds = data.bonds;
    currentMoleculeName = name; // Guardar el nombre actual

    // Actualizar nombres en los títulos
    const displayName = name.charAt(0).toUpperCase() + name.slice(1);
    moleculeName2d.textContent = displayName;
    moleculeName3d.textContent = displayName;

    // Renderizar Vista 3D
    viewer.clear();
    viewer.addModel(currentMolBlock, 'mol');
    update3DStyle();
    viewer.zoomTo();
    viewer.render();
    setup3Dinteractivity();
    toggle3DLabels();

    // Renderizar Vista 2D y ocultar overlays
    viewer2DContainer.innerHTML = data.svg;
    viewer2dOverlay.style.display = 'none';
    viewer3dOverlay.style.display = 'none';
    setup2Dinteractivity();
    setupBondInteractivity();

    // Revelar el nuevo contenido
    animatedElements.forEach(el => el.classList.remove('content-fading'));
}

export function update3DStyle() {
    if (!viewer || !currentMolBlock) return;
    const style = document.querySelector('input[name="style3d"]:checked').value;
    const showH = document.getElementById('show-hydrogens').checked;
    let styleSpec = {};

    if (style === 'line') {
        styleSpec = { line: { linewidth: 3.5 } };
    } else if (style === 'ballandstick') {
        styleSpec = { stick: { radius: 0.1 }, sphere: { scale: 0.25 } };
    } else {
        styleSpec[style] = {};
    }

    const atomSelection = showH ? {} : { elem: 'H', invert: true };
    viewer.setStyle({}, {});
    viewer.setStyle(atomSelection, styleSpec);
    if (!showH) {
        viewer.setStyle({elem: 'H'}, {});
    }
    toggle3DLabels();
    viewer.render();
}

function setup3Dinteractivity() {
    viewer.setClickable({}, true, (atom) => {
        handleAtomClick(atom);
    });
}

function setup2Dinteractivity() {
    if (panZoomInstance) {
        panZoomInstance.destroy();
        panZoomInstance = null;
    }
    const svg = viewer2DContainer.querySelector('svg');
    if (!svg) return;
    const atomIds = new Set();
    svg.querySelectorAll('[class*="atom-"]').forEach(el => {
        const match = el.getAttribute('class').match(/atom-(\d+)/);
        if (match) atomIds.add(parseInt(match[1]));
    });

    atomIds.forEach(id => {
        const atomElements = svg.querySelectorAll(`.atom-${id}`);
        const atom3D = viewer.getModel().selectedAtoms({ serial: id + 1 })[0];
        if (!atom3D) return;

        const handleMouseOver = () => {
            if (lastHoveredAtom) {
                viewer.setStyle({ serial: lastHoveredAtom.serial }, { stick: {} });
            }
            viewer.setStyle({ serial: atom3D.serial }, { stick: { color: 'yellow' } });
            lastHoveredAtom = atom3D;
            viewer.render();
        };
        const handleMouseOut = () => {
            viewer.setStyle({ serial: atom3D.serial }, { stick: {} });
            if (lastHoveredAtom && lastHoveredAtom.serial === atom3D.serial) {
                lastHoveredAtom = null;
            }
            viewer.render();
        };
        const handleClick = (event) => {
            handleAtomClick(atom3D, event);
        };
        atomElements.forEach(el => {
            el.style.cursor = 'pointer';
            el.addEventListener('mouseover', handleMouseOver);
            el.addEventListener('mouseout', handleMouseOut);
            el.addEventListener('click', handleClick);
        });
    });

    panZoomInstance = svgPanZoom(svg, {
        zoomEnabled: true,
        controlIconsEnabled: true,
        fit: true,
        center: true,
        minZoom: 0.5,
        maxZoom: 10
    });
}

function setupBondInteractivity() {
    const svg = viewer2DContainer.querySelector('svg');
    if (!svg || !currentBonds) return;
    const bondPaths = svg.querySelectorAll('path[class^="bond-"]');
    bondPaths.forEach((path, i) => {
        if (currentBonds[i]) {
            const bondData = currentBonds[i];
            path.id = `bond-${bondData.bond_index}`;
            path.classList.add('interactive-bond');
            path.addEventListener('click', (event) => handleBondClick(event, bondData));
        }
    });
}

export function toggle3DLabels() {
    if (!viewer) return;
    viewer.removeAllLabels();
    if (document.getElementById('show-atom-indices-3d').checked) {
        const atoms = viewer.getModel().selectedAtoms({});
        atoms.forEach(atom => {
            viewer.addLabel(atom.elem + (atom.serial), {
                position: { x: atom.x, y: atom.y, z: atom.z },
                fontColor: 'black',
                backgroundColor: 'lightgray',
                backgroundOpacity: 0.7
            });
        });
    }
    if (document.getElementById('show-bond-indices-3d').checked && currentBonds) {
        currentBonds.forEach(bond => {
            const atom1 = viewer.getModel().selectedAtoms({ serial: bond.atom1_index + 1 })[0];
            const atom2 = viewer.getModel().selectedAtoms({ serial: bond.atom2_index + 1 })[0];
            if (atom1 && atom2) {
                const midpoint = {
                    x: (atom1.x + atom2.x) / 2,
                    y: (atom1.y + atom2.y) / 2,
                    z: (atom1.z + atom2.z) / 2
                };
                viewer.addLabel(`E${bond.bond_index}`, {
                    position: midpoint,
                    fontColor: 'black',
                    backgroundColor: '#FFDDC1',
                    backgroundOpacity: 0.7,
                    showBackground: true
                });
            }
        });
    }
    viewer.render();
} 