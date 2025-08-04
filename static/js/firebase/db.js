import { getFirestore, collection, addDoc, getDocs } from "firebase/firestore";
import { app } from './config.js';

const db = getFirestore(app);

// Guardar un compuesto en la base de datos
async function saveCompound(compoundData) {
    try {
        const docRef = await addDoc(collection(db, "compounds"), compoundData);
        return docRef.id;
    } catch (e) {
    }
}

// Cargartodos los compuestos de la base de datos
async function loadCompounds() {
    const querySnapshot = await getDocs(collection(db, "compounds"));
    const compounds = [];
    querySnapshot.forEach((doc) => {
        compounds.push({ id: doc.id, ...doc.data() });
    });
    return compounds;
}

export { saveCompound, loadCompounds };
