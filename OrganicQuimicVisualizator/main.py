# Archivo: rdkit_gestos_corregido.py
# Sistema de visualización molecular con RDKit controlado por gestos - CORREGIDO

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import os
import mediapipe as mp
import math
import numpy as np
import base64
import io
import time
from PIL import Image, ImageDraw, ImageFont

# Importaciones de RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdMolAlign import AlignMol
    from rdkit.Chem import rdFMCS
    RDKIT_AVAILABLE = True
    print("✅ RDKit importado correctamente")
except ImportError as e:
    print(f"⚠️ RDKit no está disponible: {e}")
    print("Instalalo con: pip install rdkit")
    RDKIT_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuración de directorios
UPLOAD_FOLDER = 'static/uploads'
MOLECULES_FOLDER = 'static/molecules'
for folder in [UPLOAD_FOLDER, MOLECULES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ===== CONFIGURACIÓN DE MEDIAPIPE =====
mp_hands = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def detectar_gesto_agarre(landmarks, w, h):
    """Detecta gesto de agarre"""
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    dedos_doblados = 0
    
    for i in range(1, 5):
        tip = landmarks[tip_ids[i]]
        pip = landmarks[pip_ids[i]]
        if tip.y > pip.y:
            dedos_doblados += 1
    
    thumb_tip = landmarks[tip_ids[0]]
    thumb_pip = landmarks[pip_ids[0]]
    if thumb_tip.x > thumb_pip.x:
        dedos_doblados += 1
    
    return dedos_doblados >= 3

# ===== CLASE PRINCIPAL PARA MANEJO DE MOLÉCULAS =====
class MolecularViewer:
    def __init__(self):
        self.mol_ligando = None
        self.mol_receptor = None
        self.conformaciones = {}
        self.propiedades = {}
        
        # Estados de visualización - CORREGIDO: Iniciar en 3D
        self.viewer_state = {
            'rotacion': {'x': 0, 'y': 0, 'z': 0},
            'traslacion': {'x': 0, 'y': 0, 'z': 0},
            'zoom': 1.0,
            'modo_vista': '3d',  
            'colores': 'cpk',
            'mostrar_hidrogenos': False,
            'estilo_enlace': 'stick'
        }
        
        # Control por gestos
        self.gesto_state = {
            'modo_gesto': 'libre',
            'manos_detectadas': 0,
            'ligando_activo': False,
            'receptor_activo': False
        }
        
    def cargar_molecula_desde_smiles(self, smiles, nombre="Molecula"):
        """Carga una molécula desde SMILES - CORREGIDO"""
        if not RDKIT_AVAILABLE:
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"❌ No se pudo parsear SMILES: {smiles}")
                return None
                
            # Generar coordenadas 3D - MÉTODO CORREGIDO
            mol = Chem.AddHs(mol)
            
            # Método corregido para generar conformaciones
            try:
                # Intentar método nuevo
                AllChem.EmbedMolecule(mol, randomSeed=42)
                # Optimización corregida - sin usar OptimizeMolecule directamente
                AllChem.UFFOptimizeMolecule(mol)
            except Exception as e:
                print(f"⚠️ Error en optimización 3D, usando 2D: {e}")
                # Si falla 3D, usar solo 2D
                AllChem.Compute2DCoords(mol)
            
            # Calcular propiedades
            propiedades = self.calcular_propiedades_seguras(mol, nombre)
            
            return {
                'mol': mol,
                'propiedades': propiedades,
                'smiles': smiles
            }
            
        except Exception as e:
            print(f"❌ Error cargando molécula: {e}")
            return None
    
    def calcular_propiedades_seguras(self, mol, nombre="Molecula"):
        """Calcula propiedades moleculares con manejo de errores"""
        try:
            propiedades = {
                'nombre': nombre,
                'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'peso_molecular': round(Descriptors.MolWt(mol), 2),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds()
            }
            
            # Propiedades que pueden fallar
            try:
                propiedades['logp'] = round(Descriptors.MolLogP(mol), 2)
            except:
                propiedades['logp'] = 0.0
                
            try:
                propiedades['tpsa'] = round(Descriptors.TPSA(mol), 2)
            except:
                propiedades['tpsa'] = 0.0
                
            try:
                propiedades['hbd'] = Descriptors.NumHDonors(mol)
                propiedades['hba'] = Descriptors.NumHAcceptors(mol)
            except:
                propiedades['hbd'] = 0
                propiedades['hba'] = 0
                
            try:
                propiedades['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            except:
                propiedades['rotatable_bonds'] = 0
                
            try:
                propiedades['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            except:
                propiedades['aromatic_rings'] = 0
            
            return propiedades
            
        except Exception as e:
            print(f"⚠️ Error calculando propiedades: {e}")
            return {
                'nombre': nombre,
                'formula': 'Unknown',
                'peso_molecular': 0.0,
                'logp': 0.0,
                'tpsa': 0.0,
                'hbd': 0,
                'hba': 0,
                'rotatable_bonds': 0,
                'aromatic_rings': 0,
                'num_atoms': mol.GetNumAtoms() if mol else 0,
                'num_bonds': mol.GetNumBonds() if mol else 0
            }
    
    def generar_imagen_2d(self, mol_data, width=630, height=350):
        """Genera imagen 2D de la molécula con RDKit - CORREGIDO"""
        if not mol_data or not RDKIT_AVAILABLE:
            return self.crear_imagen_placeholder(width, height, "RDKit no disponible")
            
        try:
            mol = mol_data['mol']
            
            # Asegurar coordenadas 2D
            if not mol.GetNumConformers():
                AllChem.Compute2DCoords(mol)
            
            # Configurar drawer - MÉTODO CORREGIDO
            drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
            
            # Configuraciones del drawer
            drawer.SetFontSize(0.8)
            
            # Opciones de dibujo
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = True
            opts.addAtomIndices = False
            
            # Dibujar molécula
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Convertir a imagen
            img_data = drawer.GetDrawingText()
            
            # Convertir a base64
            img_base64 = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error generando imagen 2D: {e}")
            return self.crear_imagen_placeholder(width, height, f"Error: {str(e)[:50]}")
    
    def generar_vista_3d_simple(self, mol_data, width=630, height=350):
        """Genera vista 3D simple usando RDKit - MÉTODO MEJORADO CON ESTILOS"""
        if not mol_data or not RDKIT_AVAILABLE:
            return self.crear_imagen_placeholder(width, height, "Vista 3D no disponible")
        
        try:
            mol = mol_data['mol']
            
            # Asegurar que tiene conformación 3D
            if mol.GetNumConformers() == 0:
                # Si no tiene conformación 3D, generar una
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    print("⚠️ No se pudo optimizar, usando conformación inicial")
            
            # Manejar hidrógenos según configuración
            if not self.viewer_state['mostrar_hidrogenos']:
                # Remover hidrógenos para visualización
                mol_display = Chem.RemoveHs(mol)
            else:
                mol_display = mol
            
            # Crear una imagen 3D simulada usando proyección
            conf = mol_display.GetConformer()
            coords = []
            
            # Obtener coordenadas 3D
            for i in range(mol_display.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            coords = np.array(coords)
            
            # Aplicar rotaciones del viewer_state
            rotx = np.radians(self.viewer_state['rotacion']['x'])
            roty = np.radians(self.viewer_state['rotacion']['y'])
            rotz = np.radians(self.viewer_state['rotacion']['z'])
            
            # Matrices de rotación
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rotx), -np.sin(rotx)],
                          [0, np.sin(rotx), np.cos(rotx)]])
            
            Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                          [0, 1, 0],
                          [-np.sin(roty), 0, np.cos(roty)]])
            
            Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                          [np.sin(rotz), np.cos(rotz), 0],
                          [0, 0, 1]])
            
            # Aplicar rotaciones
            coords = coords @ Rx @ Ry @ Rz
            
            # Proyección a 2D (perspectiva simple)
            zoom = self.viewer_state['zoom']
            scale = 20 * zoom
            
            x_proj = coords[:, 0] * scale + width/2
            y_proj = coords[:, 1] * scale + height/2
            z_proj = coords[:, 2]  # Para profundidad
            
            # Crear imagen PIL
            img = Image.new('RGB', (width, height), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)
            
            # Estilo de renderizado según configuración
            estilo = self.viewer_state['estilo_enlace']
            
            # Dibujar enlaces (si no es estilo sphere)
            if estilo != 'sphere':
                for bond in mol_display.GetBonds():
                    atom1_idx = bond.GetBeginAtomIdx()
                    atom2_idx = bond.GetEndAtomIdx()
                    
                    x1, y1 = int(x_proj[atom1_idx]), int(y_proj[atom1_idx])
                    x2, y2 = int(x_proj[atom2_idx]), int(y_proj[atom2_idx])
                    
                    # Color del enlace basado en profundidad
                    depth_avg = (z_proj[atom1_idx] + z_proj[atom2_idx]) / 2
                    intensity = int(255 * (0.5 + depth_avg * 0.1))
                    intensity = max(100, min(255, intensity))
                    
                    # Grosor según estilo
                    if estilo == 'stick':
                        line_width = 3
                    elif estilo == 'line':
                        line_width = 1
                    else:
                        line_width = 2
                    
                    draw.line([(x1, y1), (x2, y2)], fill=(intensity, intensity, intensity), width=line_width)
            
            # Dibujar átomos
            for i in range(mol_display.GetNumAtoms()):
                atom = mol_display.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                
                x, y = int(x_proj[i]), int(y_proj[i])
                
                # Color por elemento (CPK standard)
                if symbol == 'C':
                    color = (64, 64, 64)  # Gris oscuro
                elif symbol == 'O':
                    color = (255, 50, 50)  # Rojo
                elif symbol == 'N':
                    color = (50, 50, 255)  # Azul
                elif symbol == 'S':
                    color = (255, 255, 50)  # Amarillo
                elif symbol == 'P':
                    color = (255, 165, 0)  # Naranja
                elif symbol == 'F':
                    color = (144, 224, 80)  # Verde claro
                elif symbol == 'Cl':
                    color = (31, 240, 31)  # Verde
                elif symbol == 'Br':
                    color = (166, 41, 41)  # Marrón
                elif symbol == 'H':
                    color = (255, 255, 255)  # Blanco
                else:
                    color = (200, 200, 200)  # Gris claro
                
                # Tamaño basado en profundidad y estilo
                depth_factor = 1 + z_proj[i] * 0.1
                
                if estilo == 'sphere':
                    # Esferas grandes
                    radius = max(8, int(12 * depth_factor * zoom))
                elif estilo == 'stick':
                    # Esferas medianas
                    radius = max(4, int(8 * depth_factor * zoom))
                else:  # line
                    # Esferas pequeñas
                    radius = max(2, int(4 * depth_factor * zoom))
                
                # Dibujar átomo
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=(255,255,255), width=1)
                
                # Etiqueta del átomo
                show_label = False
                if estilo == 'line':
                    # En modo line, mostrar todas las etiquetas
                    show_label = True
                elif symbol != 'C':
                    # Mostrar etiquetas de elementos no-carbono
                    show_label = True
                elif atom.GetDegree() <= 1:
                    # Mostrar etiquetas de átomos terminales
                    show_label = True
                
                if show_label and radius > 3:
                    # Calcular posición del texto
                    text_x = x - len(symbol) * 3
                    text_y = y - 6
                    
                    # Fondo para el texto
                    if estilo == 'line':
                        draw.rectangle([text_x-2, text_y-2, text_x+len(symbol)*6+2, text_y+10], 
                                     fill=(0,0,0,128), outline=(255,255,255))
                    
                    draw.text((text_x, text_y), symbol, fill=(255, 255, 255))
            
            # Información del estilo en la esquina
            info_text = f"Estilo: {estilo.capitalize()}"
            if self.viewer_state['mostrar_hidrogenos']:
                info_text += " + H"
            
            draw.text((10, height-25), info_text, fill=(255, 255, 255))
            
            # Convertir a base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error generando vista 3D: {e}")
            return self.crear_imagen_placeholder(width, height, f"Error 3D: {str(e)[:30]}")
    
    def crear_imagen_placeholder(self, width=630, height=350, texto="Sin molécula"):
        """Crea imagen placeholder cuando no se puede renderizar"""
        try:
            # Crear imagen PIL
            img = Image.new('RGB', (width, height), color=(30, 30, 50))
            draw = ImageDraw.Draw(img)
            
            # Dibujar texto centrado
            try:
                # Intentar usar fuente por defecto
                font_size = 16
                bbox = draw.textbbox((0, 0), texto)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                draw.text((x, y), texto, fill=(255, 255, 255))
            except:
                # Si falla, texto simple
                draw.text((width//4, height//2), texto, fill=(255, 255, 255))
            
            # Convertir a base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error creando placeholder: {e}")
            return None
    
    def calcular_docking_score(self, mol1_data, mol2_data):
        """Calcula score de docking simplificado entre dos moléculas"""
        if not mol1_data or not mol2_data or not RDKIT_AVAILABLE:
            return 0.0
            
        try:
            mol1 = mol1_data['mol']
            mol2 = mol2_data['mol']
            
            # Score basado en propiedades moleculares
            prop1 = mol1_data['propiedades']
            prop2 = mol2_data['propiedades']
            
            # Factores de complementariedad
            hbd_hba_complement = min(prop1['hbd'], prop2['hba']) + min(prop1['hba'], prop2['hbd'])
            size_complement = 50 - abs(prop1['peso_molecular'] - prop2['peso_molecular']) / 10
            logp_complement = 25 - abs(prop1['logp'] - prop2['logp']) * 5
            
            final_score = max(0, hbd_hba_complement * 10 + size_complement + logp_complement)
            
            return min(100, max(-50, final_score))
            
        except Exception as e:
            print(f"❌ Error calculando docking score: {e}")
            return 0.0

# Instancia global del visualizador
molecular_viewer = MolecularViewer()

# ===== DEFINICIÓN DE MOLÉCULAS PREDEFINIDAS =====
MOLECULAS_PREDEFINIDAS = {
    'estradiol': {
        'smiles': 'C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O',
        'nombre': 'Estradiol',
        'descripcion': 'Hormona sexual femenina'
    },
    'fulvestrant': {
        'smiles': 'C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@H](C(F)(F)C(F)(F)S(=O)CCCCCCCCC)C2',
        'nombre': 'Fulvestrant',
        'descripcion': 'Medicamento'
    }
}

# ===== RESTO DEL CÓDIGO IGUAL =====
gestos_estado = {
    'modelo_estado': {
        'rotacion': {'x': 0, 'y': 0, 'z': 0},
        'posicion': {'x': 0, 'y': 0, 'z': 0},
        'escala': 1.0,
        'modo_gesto': 'libre',
        'manos_detectadas': 0
    },
    'molecular_data': {
        'ligando': None,
        'receptor': None,
        'docking_score': 0.0,
        'propiedades_ligando': {},
        'propiedades_receptor': {}
    }
}

ultimo_envio_websocket = 0
clientes_conectados = 0

def procesar_gestos_molecular(frame):
    """Procesa gestos y actualiza el estado molecular - CORREGIDO"""
    global gestos_estado, ultimo_envio_websocket, clientes_conectados
    
    frame_flip = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)
    
    h, w, _ = frame_flip.shape
    
    # INICIALIZAR VARIABLE - CORRECCIÓN DEL ERROR
    num_manos = 0
    
    # Resetear estado
    gestos_estado['modelo_estado']['manos_detectadas'] = 0
    
    if resultado.multi_hand_landmarks:
        num_manos = len(resultado.multi_hand_landmarks)
        gestos_estado['modelo_estado']['manos_detectadas'] = num_manos
        
        # Dibujar manos
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame_flip, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # ===== LÓGICA DE GESTOS =====
        if num_manos == 1:
            hand = resultado.multi_hand_landmarks[0]
            palm = hand.landmark[9]
            palm_x, palm_y = palm.x, palm.y
            
            if detectar_gesto_agarre(hand.landmark, w, h):
                gestos_estado['modelo_estado']['modo_gesto'] = "rotacion_molecular"
                
                # Convertir posición a rotación
                molecular_viewer.viewer_state['rotacion']['y'] = (palm_x - 0.5) * 720
                molecular_viewer.viewer_state['rotacion']['x'] = (palm_y - 0.5) * 360
                
                gestos_estado['modelo_estado']['rotacion'] = molecular_viewer.viewer_state['rotacion']
                
            else:
                gestos_estado['modelo_estado']['modo_gesto'] = "libre"
        
        elif num_manos == 2:
            hand1, hand2 = resultado.multi_hand_landmarks[0], resultado.multi_hand_landmarks[1]
            palm1, palm2 = hand1.landmark[9], hand2.landmark[9]
            
            # Calcular distancia entre manos
            dist = math.sqrt((palm1.x - palm2.x)**2 + (palm1.y - palm2.y)**2)
            
            agarre1 = detectar_gesto_agarre(hand1.landmark, w, h)
            agarre2 = detectar_gesto_agarre(hand2.landmark, w, h)
            
            if agarre1 and agarre2:
                gestos_estado['modelo_estado']['modo_gesto'] = "traslacion_molecular"
                
                # Mover molécula en 3D
                centro_x = (palm1.x + palm2.x) / 2
                centro_y = (palm1.y + palm2.y) / 2
                
                molecular_viewer.viewer_state['traslacion']['x'] = (centro_x - 0.5) * 4
                molecular_viewer.viewer_state['traslacion']['y'] = (0.5 - centro_y) * 4
                
            elif dist > 0.3:
                gestos_estado['modelo_estado']['modo_gesto'] = "zoom_molecular"
                
                # Zoom basado en distancia entre manos
                zoom_factor = min(max(dist / 0.4, 0.3), 3.0)
                molecular_viewer.viewer_state['zoom'] = zoom_factor
                gestos_estado['modelo_estado']['escala'] = zoom_factor
                
                # Dibujar línea entre manos
                x1, y1 = int(palm1.x * w), int(palm1.y * h)
                x2, y2 = int(palm2.x * w), int(palm2.y * h)
                cv2.line(frame_flip, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
            else:
                gestos_estado['modelo_estado']['modo_gesto'] = "cambio_vista"
        
        else:
            gestos_estado['modelo_estado']['modo_gesto'] = "libre"
    else:
        gestos_estado['modelo_estado']['modo_gesto'] = "libre"
    
    # Calcular docking score si hay dos moléculas
    if molecular_viewer.mol_ligando and molecular_viewer.mol_receptor:
        score = molecular_viewer.calcular_docking_score(
            molecular_viewer.mol_ligando, 
            molecular_viewer.mol_receptor
        )
        gestos_estado['molecular_data']['docking_score'] = score
    
    # Dibujar información en frame
    cv2.putText(frame_flip, f"Manos: {num_manos}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_flip, f"Modo: {gestos_estado['modelo_estado']['modo_gesto']}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if gestos_estado['molecular_data']['docking_score'] > 0:
        cv2.putText(frame_flip, f"Score: {gestos_estado['molecular_data']['docking_score']:.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Estado de RDKit
    rdkit_status = "RDKit OK" if RDKIT_AVAILABLE else "RDKit NO"
    cv2.putText(frame_flip, rdkit_status, (10, h-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if RDKIT_AVAILABLE else (0, 0, 255), 2)
    
    # Estado del modo de vista
    modo_vista = molecular_viewer.viewer_state['modo_vista'].upper()
    cv2.putText(frame_flip, f"Vista: {modo_vista}", (10, h-60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Enviar datos via WebSocket
    tiempo_actual = time.time()
    if clientes_conectados > 0 and tiempo_actual - ultimo_envio_websocket > 0.1:
        try:
            socketio.emit('molecular_update', gestos_estado)
            ultimo_envio_websocket = tiempo_actual
        except Exception as e:
            print(f"❌ Error emitiendo WebSocket molecular: {e}")
    
    return frame_flip

def generar_frames_molecular():
    """Generador de frames con procesamiento molecular - CORREGIDO"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la cámara")
        # Intentar con diferentes índices
        for i in range(1, 4):
            print(f"🔄 Intentando cámara índice {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Cámara encontrada en índice {i}")
                break
        else:
            print("❌ No se encontró ninguna cámara disponible")
            return
    
    print("✅ Cámara abierta para visualización molecular")
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Error leyendo frame {frame_count}")
            break
        
        try:
            frame_procesado = procesar_gestos_molecular(frame)
            
            _, buffer = cv2.imencode('.jpg', frame_procesado)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            
        except Exception as e:
            print(f"❌ Error procesando frame {frame_count}: {e}")
            # Continuar sin procesar gestos
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    print("🔄 Cámara liberada")

# ===== RUTAS FLASK =====

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camara')
def camara():
    return Response(generar_frames_molecular(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cargar_molecula', methods=['POST'])
def cargar_molecula():
    """Cargar molécula desde SMILES o nombre predefinido - CORREGIDO"""
    
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit no está disponible. Instálalo con: pip install rdkit'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        tipo = data.get('tipo', 'ligando')  # 'ligando' o 'receptor'
        
        if 'smiles' in data:
            # Cargar desde SMILES
            mol_data = molecular_viewer.cargar_molecula_desde_smiles(
                data['smiles'], 
                data.get('nombre', 'Molecula personalizada')
            )
        elif 'predefinida' in data:
            # Cargar molécula predefinida
            mol_info = MOLECULAS_PREDEFINIDAS.get(data['predefinida'])
            if not mol_info:
                return jsonify({'error': 'Molécula predefinida no encontrada'}), 400
            
            mol_data = molecular_viewer.cargar_molecula_desde_smiles(
                mol_info['smiles'],
                mol_info['nombre']
            )
        else:
            return jsonify({'error': 'Debe proporcionar SMILES o nombre predefinido'}), 400
        
        if not mol_data:
            return jsonify({'error': 'No se pudo cargar la molécula. Verifica el SMILES.'}), 400
        
        # Asignar molécula
        if tipo == 'ligando':
            molecular_viewer.mol_ligando = mol_data
            gestos_estado['molecular_data']['propiedades_ligando'] = mol_data['propiedades']
        else:
            molecular_viewer.mol_receptor = mol_data
            gestos_estado['molecular_data']['propiedades_receptor'] = mol_data['propiedades']
        
        # Generar imagen según modo actual
        if molecular_viewer.viewer_state['modo_vista'] == '2d':
            imagen = molecular_viewer.generar_imagen_2d(mol_data)
        else:
            imagen = molecular_viewer.generar_vista_3d_simple(mol_data)
        
        print(f"✅ {tipo} cargado: {mol_data['propiedades']['nombre']}")
        
        return jsonify({
            'success': True,
            'propiedades': mol_data['propiedades'],
            'smiles': mol_data['smiles'],
            'imagen_2d': imagen,  # Nombre mantenido para compatibilidad
            'tipo': tipo
        })
        
    except Exception as e:
        error_msg = f"Error cargando molécula: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/generar_imagen')
def generar_imagen():
    """Generar imagen actualizada de la molécula - CORREGIDO CON 3D"""
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit no disponible'}), 500
    
    tipo = request.args.get('tipo', 'ligando')
    modo = request.args.get('modo', molecular_viewer.viewer_state['modo_vista'])
    
    mol_data = molecular_viewer.mol_ligando if tipo == 'ligando' else molecular_viewer.mol_receptor
    
    if not mol_data:
        return jsonify({'error': f'No hay {tipo} cargado'}), 400
    
    try:
        # Generar imagen según el modo
        if modo == '2d':
            imagen = molecular_viewer.generar_imagen_2d(mol_data)
        else:  # modo == '3d'
            imagen = molecular_viewer.generar_vista_3d_simple(mol_data)
        
        if not imagen:
            return jsonify({'error': 'No se pudo generar la imagen'}), 500
        
        return jsonify({
            'success': True,
            'imagen': imagen,
            'viewer_state': molecular_viewer.viewer_state,
            'modo': modo
        })
        
    except Exception as e:
        error_msg = f"Error generando imagen: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/cambiar_modo_vista', methods=['POST'])
def cambiar_modo_vista():
    """Cambiar entre modo 2D y 3D - NUEVA RUTA"""
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit no disponible'}), 500
    
    try:
        data = request.get_json()
        nuevo_modo = data.get('modo', '2d')
        
        if nuevo_modo not in ['2d', '3d']:
            return jsonify({'error': 'Modo inválido. Use "2d" o "3d"'}), 400
        
        # Actualizar modo
        molecular_viewer.viewer_state['modo_vista'] = nuevo_modo
        
        print(f"🎨 Modo de vista cambiado a: {nuevo_modo}")
        
        return jsonify({
            'success': True,
            'modo': nuevo_modo,
            'viewer_state': molecular_viewer.viewer_state
        })
        
    except Exception as e:
        error_msg = f"Error cambiando modo: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/moleculas_predefinidas')
def get_moleculas_predefinidas():
    """Obtener lista de moléculas predefinidas"""
    return jsonify(MOLECULAS_PREDEFINIDAS)

@app.route('/api/reset_viewer')
def reset_viewer():
    """Reset del visualizador molecular"""
    global molecular_viewer, gestos_estado
    
    try:
        # Reset del viewer - MANTENER MODO 3D POR DEFECTO
        molecular_viewer.viewer_state = {
            'rotacion': {'x': 0, 'y': 0, 'z': 0},
            'traslacion': {'x': 0, 'y': 0, 'z': 0},
            'zoom': 1.0,
            'modo_vista': '3d',  # Mantener 3D como defecto
            'colores': 'cpk',
            'mostrar_hidrogenos': False,
            'estilo_enlace': 'stick'
        }
        
        # Reset de moléculas
        molecular_viewer.mol_ligando = None
        molecular_viewer.mol_receptor = None
        
        # Reset del estado de gestos
        gestos_estado['modelo_estado'] = {
            'rotacion': {'x': 0, 'y': 0, 'z': 0},
            'posicion': {'x': 0, 'y': 0, 'z': 0},
            'escala': 1.0,
            'modo_gesto': 'libre',
            'manos_detectadas': 0
        }
        
        gestos_estado['molecular_data'] = {
            'ligando': None,
            'receptor': None,
            'docking_score': 0.0,
            'propiedades_ligando': {},
            'propiedades_receptor': {}
        }
        
        print("🔄 Viewer reseteado correctamente")
        return jsonify({'success': True, 'modo_vista': '3d'})
        
    except Exception as e:
        error_msg = f"Error reseteando viewer: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

# ===== WEBSOCKET EVENTS =====

@socketio.on('connect')
def handle_connect():
    global clientes_conectados
    clientes_conectados += 1
    print(f'✅ Cliente molecular conectado. Total: {clientes_conectados}')
    emit('molecular_update', gestos_estado)

@socketio.on('disconnect') 
def handle_disconnect():
    global clientes_conectados
    clientes_conectados = max(0, clientes_conectados - 1)
    print(f'❌ Cliente molecular desconectado. Total: {clientes_conectados}')

@socketio.on('cambiar_estilo')
def handle_cambiar_estilo(data):
    """Cambiar estilo de visualización - MEJORADO"""
    try:
        actualizado = False
        
        if 'estilo_enlace' in data:
            molecular_viewer.viewer_state['estilo_enlace'] = data['estilo_enlace']
            actualizado = True
            print(f"🎨 Estilo de enlace cambiado a: {data['estilo_enlace']}")
            
        if 'colores' in data:
            molecular_viewer.viewer_state['colores'] = data['colores']
            actualizado = True
            print(f"🎨 Esquema de colores cambiado a: {data['colores']}")
            
        if 'mostrar_hidrogenos' in data:
            molecular_viewer.viewer_state['mostrar_hidrogenos'] = data['mostrar_hidrogenos']
            actualizado = True
            estado_h = "mostrar" if data['mostrar_hidrogenos'] else "ocultar"
            print(f"🎨 Hidrógenos: {estado_h}")
        
        if actualizado:
            # Emitir estado actualizado a todos los clientes
            emit('viewer_state_update', molecular_viewer.viewer_state, broadcast=True)
            print(f"📡 Estado del viewer actualizado y enviado a clientes")
        
    except Exception as e:
        print(f"❌ Error cambiando estilo: {e}")

@socketio.on('regenerar_imagen')
def handle_regenerar_imagen(data):
    """Regenerar imagen con estilos actualizados - NUEVA FUNCIÓN"""
    try:
        tipo = data.get('tipo', 'ligando')
        
        # Indicar que se debe regenerar la imagen
        emit('regenerar_molecula', {'tipo': tipo}, broadcast=True)
        print(f"🔄 Solicitando regeneración de imagen para {tipo}")
        
    except Exception as e:
        print(f"❌ Error regenerando imagen: {e}")

# ===== RUTA PARA FAVICON (EVITAR ERROR 404) =====
@app.route('/favicon.ico')
def favicon():
    return '', 204

# ===== FUNCIÓN DE TESTING =====
@app.route('/test')
def test_rdkit():
    """Endpoint para probar RDKit"""
    if not RDKIT_AVAILABLE:
        return jsonify({
            'rdkit_available': False,
            'error': 'RDKit no está instalado',
            'install_command': 'pip install rdkit'
        })
    
    try:
        # Test básico de RDKit
        test_smiles = 'CCO'  # Etanol
        mol = Chem.MolFromSmiles(test_smiles)
        
        if mol is None:
            return jsonify({
                'rdkit_available': True,
                'test_passed': False,
                'error': 'No se pudo crear molécula de prueba'
            })
        
        # Test de propiedades
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Test de imagen 2D
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(200, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        img_base64 = base64.b64encode(img_data).decode()
        
        return jsonify({
            'rdkit_available': True,
            'test_passed': True,
            'test_molecule': 'Etanol (CCO)',
            'molecular_weight': round(mw, 2),
            'logp': round(logp, 2),
            'image_generated': True,
            'image_size': len(img_base64),
            'camera_available': cv2.VideoCapture(0).isOpened(),
            'viewer_state': molecular_viewer.viewer_state
        })
        
    except Exception as e:
        return jsonify({
            'rdkit_available': True,
            'test_passed': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🧪 Iniciando visualizador molecular con RDKit...")
    print("=" * 50)
    
    # Verificar RDKit
    if not RDKIT_AVAILABLE:
        print("❌ RDKit no está disponible. Por favor instálalo:")
        print("   OPCIÓN 1: pip install rdkit")
        print("   OPCIÓN 2: conda install -c conda-forge rdkit")
        print("   OPCIÓN 3: pip install rdkit-pypi")
        print("")
    else:
        print("✅ RDKit disponible")
        print("📋 Moléculas predefinidas disponibles:")
        for key, mol in MOLECULAS_PREDEFINIDAS.items():
            print(f"   - {key}: {mol['nombre']}")
        print("")
    
    # Verificar cámara
    cap_test = cv2.VideoCapture(0)
    if cap_test.isOpened():
        print("✅ Cámara disponible")
        cap_test.release()
    else:
        print("⚠️ Cámara no detectada - el sistema funcionará sin gestos")
    
    print("=" * 50)
    print("🌐 Servidor iniciando en http://localhost:5000")
    print("🔬 Endpoint de prueba: http://localhost:5000/test")
    print("📹 Stream de cámara: http://localhost:5000/camara")
    print("🎨 Modo por defecto: Vista 3D")
    print("=" * 50)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)