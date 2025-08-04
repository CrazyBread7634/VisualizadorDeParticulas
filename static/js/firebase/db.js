import { getFirestore, collection, addDoc, getDocs, deleteDoc, doc } from "firebase/firestore";
import { app } from './config.js';

const db = getFirestore(app);

// Guardar un compuesto en la base de datos
async function saveCompound(compoundData) {
    try {
        const docRef = await addDoc(collection(db, "compounds"), compoundData);
        return docRef.id;
    } catch (e) {
        console.error("Error adding document: ", e);
        throw e;
    }
}

// Cargar todos los compuestos de la base de datos
async function loadCompounds() {
    try {
        const querySnapshot = await getDocs(collection(db, "compounds"));
        const compounds = [];
        querySnapshot.forEach((doc) => {
            compounds.push({ id: doc.id, ...doc.data() });
        });
        return compounds;
    } catch (e) {
        console.error("Error loading compounds: ", e);
        return [];
    }
}

// Eliminar un compuesto de la base de datos
async function deleteCompound(compoundId) {
    try {
        await deleteDoc(doc(db, "compounds", compoundId));
        return true;
    } catch (e) {
        console.error("Error deleting document: ", e);
        throw e;
    }
}

export { saveCompound, loadCompounds, deleteCompound };
