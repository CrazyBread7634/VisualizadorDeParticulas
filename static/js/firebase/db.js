import { getFirestore, collection, addDoc, getDocs, deleteDoc, doc } from "firebase/firestore";
import { app } from './config.js';

const db = getFirestore(app);

async function saveCompound(compoundData) {
    try {
        const docRef = await addDoc(collection(db, "compounds"), compoundData);
        return docRef.id;
    } catch (e) {
    }
}

async function loadCompounds() {
    try {
        const querySnapshot = await getDocs(collection(db, "compounds"));
        const compounds = [];
        querySnapshot.forEach((doc) => {
            compounds.push({ id: doc.id, ...doc.data() });
        });
        return compounds;
    } catch (e) {
    }
}

async function deleteCompound(compoundId) {
    try {
        await deleteDoc(doc(db, "compounds", compoundId));
        return true;
    } catch (e) {
    }
}

export { saveCompound, loadCompounds, deleteCompound };
