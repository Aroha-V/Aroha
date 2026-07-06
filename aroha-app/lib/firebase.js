// Firebase CLIENT SDK — runs in the browser
// Values are read from NEXT_PUBLIC_* env vars (safe to expose to client)

import { initializeApp, getApps } from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';

const firebaseConfig = {
  apiKey:     process.env.NEXT_PUBLIC_FIREBASE_API_KEY?.replace(/['"]/g, ''),
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN?.replace(/['"]/g, ''),
  projectId:  process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID?.replace(/['"]/g, ''),
  appId:      process.env.NEXT_PUBLIC_FIREBASE_APP_ID?.replace(/['"]/g, ''),
};

if (typeof window !== 'undefined') {
  console.log('🔥 [Firebase debug] Config details:', {
    apiKey: firebaseConfig.apiKey ? `${firebaseConfig.apiKey.substring(0, 7)}... (length: ${firebaseConfig.apiKey.length})` : 'undefined',
    projectId: firebaseConfig.projectId || 'undefined',
    appId: firebaseConfig.appId ? `${firebaseConfig.appId.substring(0, 10)}...` : 'undefined',
  });
}

// Prevent duplicate app initialisation on hot-reloads
const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

/**
 * Open a Google sign-in popup and return the Firebase ID token.
 * The ID token is sent to /api/auth/google for server-side verification.
 */
export async function signInWithGoogle() {
  const result = await signInWithPopup(auth, googleProvider);
  const idToken = await result.user.getIdToken();
  return { idToken, user: result.user };
}

/** Sign out of Firebase (client-side only) */
export async function firebaseSignOut() {
  await signOut(auth);
}
