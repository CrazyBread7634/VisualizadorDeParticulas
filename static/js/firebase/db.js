import { getFirestore, collection, addDoc, getDocs } from "firebase/firestore";
import { app } from './config.js';

const db = getFirestore(app);

// Guarda un compuesto en la base de datos
async function saveCompound(compoundData) {
    try {
        const docRef = await addDoc(collection(db, "compounds"), compoundData);
        console.log("Document written with ID: ", docRef.id);
        return docRef.id;
    } catch (e) {
        console.error("Error adding document: ", e);
    }
}

// Carga todos los compuestos de la base de datos
async function loadCompounds() {
    const querySnapshot = await getDocs(collection(db, "compounds"));
    const compounds = [];
    querySnapshot.forEach((doc) => {
        compounds.push({ id: doc.id, ...doc.data() });
    });
    return compounds;
}

export { saveCompound, loadCompounds };
